/*
 * SPDX-FileCopyrightText: Copyright (c) 2019 NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution. Neither the name of
 * the copyright holder nor the names of its contributors may be used to
 * endorse or promote products derived from this software without specific
 * prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

#pragma once

#include <assert.h>
#include <atomic>
#include <cstring>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

// Prevent cuda::atomic_ref becoming an indirect call in device code. This is
// the same CUDA compatibility define used by NVBit 1.8.1's channel.hpp.
#define _CCCL_ATOMIC_UNSAFE_AUTOMATIC_STORAGE

#include <cuda/atomic>
#include <cuda_runtime.h>

#include "utils.h"

/*
 * NVBit channel synchronization compatible with CUDA 13.2 injected code.
 *
 * The public NVBit 1.8 pthread API is intentionally retained here. In
 * particular, this file retains the pthread-based registration surface; the
 * separate 1.8.1 thread-registration rename is out of scope for this patch.
 *
 * The device channel uses a byte reservation counter (head_bytes), a byte
 * publication counter (tail_bytes). Producers fence payload writes before
 * publishing completion. Atomic polling prevents device counter loads from
 * being hoisted out of wait loops. The mapped host doorbell uses volatile
 * device accesses with a system fence before notification; host-side atomic
 * publication and the NVBit 1.8 pthread interface are retained.
 */
class ChannelDev {
  private:
    int id{0};

    // Mapped pinned host memory visible to both CPU and GPU.
    int* doorbell{nullptr};

    // With USE_ASYNC_STREAM this is device memory; otherwise it is mapped
    // pinned host memory. The pointer is passed to the device channel during
    // init() and is therefore valid in either mode.
    uint8_t* buff{nullptr};
    uint32_t buff_size{0};

    // Reservation/publication counters in bytes from buff's base.  A single
    // producer lock makes reservation and publication one ordered operation;
    // without it, two warps can publish out of order and a rollover flusher
    // can wait forever for an exact prefix length.
    alignas(8) unsigned long long head_bytes{0};
    alignas(8) unsigned long long tail_bytes{0};
    unsigned int push_lock{0};

  public:
    ChannelDev() = default;

    // Use the original NVBit channel device protocol here. CUDA 13.2 can
    // compile cuda::atomic_ref for an ordinary SM120 kernel, but NVBit's
    // device-function linker rejects the generated atomic_ref image with
    // CUDA_ERROR_INVALID_SOURCE. The legacy atomicAdd/fence protocol is the
    // NVBit-compatible path exercised by the SM120 capture workloads.
    __device__ __forceinline__ void push(const void* packet, uint32_t nbytes) {
        if (nbytes == 0 || nbytes > buff_size) {
            return;
        }

        while (atomicCAS(&push_lock, 0U, 1U) != 0U) {
        }

        if (head_bytes + nbytes > buff_size) {
            flush_bytes(tail_bytes);
        }

        const unsigned long long curr = head_bytes;
        memcpy(buff + curr, packet, nbytes);
        __threadfence_system();
        head_bytes = curr + nbytes;
        tail_bytes = head_bytes;
        __threadfence();
        atomicExch(&push_lock, 0U);
    }

    __device__ __forceinline__ void flush() {
        // Called by the stream-ordered flush kernel after producers finish.
        while (atomicCAS(&push_lock, 0U, 1U) != 0U) {
        }
        flush_bytes(tail_bytes);
        __threadfence();
        atomicExch(&push_lock, 0U);
    }

  private:
    __device__ __forceinline__ void flush_bytes(unsigned long long nbytes) {
        if (nbytes == 0) {
            return;
        }

        // Ensure every reservation in the sealed prefix has published its
        // payload before notifying the CPU.
        __threadfence_system();
        volatile int* db = doorbell;
        *db = static_cast<int>(nbytes);

        // The host drains the announced prefix and release-stores zero.
        while (*db != 0) {
        }

        tail_bytes = 0;
        __threadfence();
        head_bytes = 0;
    }

    /* called by the ChannelHost init */
    void init(int new_id, int* host_doorbell_ptr, uint8_t* buff_ptr,
              uint32_t new_buff_size) {
        id = new_id;
        doorbell = host_doorbell_ptr;
        buff = buff_ptr;
        buff_size = new_buff_size;
        head_bytes = 0;
        tail_bytes = 0;
        push_lock = 0;
    }

    friend class ChannelHost;
};

class ChannelHost {
  private:
    int* doorbell{nullptr};

    cudaStream_t stream{0};
    ChannelDev* ch_dev{nullptr};

    // The host-side read cursor is a byte offset into the current announced
    // prefix. It replaces the old raw-pointer cursor while preserving the
    // existing recv() API.
    uint8_t* buff{nullptr};
    uint32_t read_head_bytes{0};

    /* receiving thread: retain NVBit 1.8's pthread API intentionally */
    pthread_t thread{};
    std::atomic<bool> thread_started{false};

    static int load_doorbell_acquire(const int* p) {
        return __atomic_load_n(p, __ATOMIC_ACQUIRE);
    }

    static void store_doorbell_release(int* p, int value) {
        __atomic_store_n(p, value, __ATOMIC_RELEASE);
    }

  public:
    int id{0};
    int buff_size{0};

  public:
    ChannelHost() = default;

    void init(int new_id, int new_buff_size, ChannelDev* ch_dev_in,
              void* (*thread_fun)(void*), void* args = NULL) {
        id = new_id;
        buff_size = new_buff_size;
        ch_dev = ch_dev_in;
        read_head_bytes = 0;

        /* get device properties */
        cudaDeviceProp prop;
        int device = 0;
        cudaGetDeviceProperties(&prop, device);
        if (prop.canMapHostMemory == 0) {
            CUDA_SAFECALL(cudaSetDeviceFlags(cudaDeviceMapHost));
        }

#ifdef USE_ASYNC_STREAM
        int priority_high = 0;
        int priority_low = 0;
        CUDA_SAFECALL(
            cudaDeviceGetStreamPriorityRange(&priority_low, &priority_high));
        CUDA_SAFECALL(cudaStreamCreateWithPriority(
            &stream, cudaStreamNonBlocking, priority_high));
#endif

        CUDA_SAFECALL(cudaHostAlloc(reinterpret_cast<void**>(&doorbell),
                                    sizeof(int), cudaHostAllocMapped));
        *doorbell = 0;

#ifdef USE_ASYNC_STREAM
        CUDA_SAFECALL(cudaMalloc(reinterpret_cast<void**>(&buff), buff_size));
#else
        // Explicitly request a mapped host buffer so the device-side pointer
        // and the CPU reader observe the same storage under the atomic
        // doorbell protocol.
        CUDA_SAFECALL(cudaHostAlloc(reinterpret_cast<void**>(&buff),
                                    buff_size, cudaHostAllocMapped));
#endif

        ch_dev->init(id, doorbell, buff, static_cast<uint32_t>(buff_size));

        if (thread_fun != NULL) {
            thread_started.store(true, std::memory_order_release);
            pthread_create(&thread, NULL, (void* (*)(void*))thread_fun, args);
        } else {
            thread_started.store(false, std::memory_order_release);
        }
    }

    /* when used in nvbit we don't want to dealloc because
     * when modules are unloaded the driver automatically deallocates CUDA
     * malloc, so further deallocs done here will result in errors */
    void destroy(bool dealloc) {
        if (thread_started.load(std::memory_order_acquire)) {
            thread_started.store(false, std::memory_order_release);
            pthread_join(thread, NULL);
        }
        if (dealloc) {
            CUDA_SAFECALL(cudaFreeHost((int*)doorbell));
#ifdef USE_ASYNC_STREAM
            CUDA_SAFECALL(cudaStreamDestroy(stream));
            CUDA_SAFECALL(cudaFree(buff));
#else
            CUDA_SAFECALL(cudaFreeHost(buff));
#endif
        }
    }

    bool is_active() const {
        return thread_started.load(std::memory_order_acquire);
    }

    uint32_t recv(void* out, uint32_t max_out_size) {
        assert(max_out_size > 0);
        assert(doorbell != NULL);

        const int announced = load_doorbell_acquire(doorbell);
        if (announced == 0) {
            return 0;
        }

        const uint32_t total_nbytes = static_cast<uint32_t>(announced);
        const uint32_t nbytes =
            (total_nbytes > max_out_size) ? max_out_size : total_nbytes;

#ifdef USE_ASYNC_STREAM
        CUDA_SAFECALL(cudaMemcpyAsync(out, buff + read_head_bytes, nbytes,
                                      cudaMemcpyDeviceToHost, stream));
        CUDA_SAFECALL(cudaStreamSynchronize(stream));
#else
        memcpy(out, buff + read_head_bytes, nbytes);
#endif

        const uint32_t bytes_left = total_nbytes - nbytes;
        read_head_bytes = (bytes_left == 0) ? 0 : (read_head_bytes + nbytes);

        store_doorbell_release(doorbell, static_cast<int>(bytes_left));
        return nbytes;
    }

    /* Retained NVBit 1.8 API; do not rename as part of this patch. */
    pthread_t get_thread() { return thread; }

    friend class MultiChannelHost;
};
