#include "nvbit_tool.h"
#include "nvbit.h"
#include "capture.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

static Capture *capture = nullptr;
static thread_local bool internal = false;
static bool launched = false;
static unsigned nregs = 0;
static void check(cudaError_t error) {
    if (error != cudaSuccess) { fprintf(stderr, "capture CUDA: %s\n", cudaGetErrorString(error)); std::exit(2); }
}
void nvbit_tool_init(CUcontext ctx) {
    (void)ctx;
    internal = true;
    if (capture) { fprintf(stderr, "multiple contexts unsupported\n"); std::exit(2); }
    check(cudaMallocManaged(&capture, sizeof(Capture)));
    internal = false;
}
void nvbit_at_cuda_event(CUcontext ctx, int is_exit, nvbit_api_cuda_t cbid,
                         const char *, void *parameters, CUresult *) {
    if (internal || cbid != API_CUDA_cuLaunchKernel) return;
    internal = true;
    auto *p = static_cast<cuLaunchKernel_params *>(parameters);
    if (!is_exit) {
        if (launched || !capture || p->gridDimX != 1 || p->gridDimY != 1 || p->gridDimZ != 1 ||
            p->blockDimX != kLanes || p->blockDimY != 1 || p->blockDimZ != 1 ||
            std::strcmp(nvbit_get_func_name(ctx, p->f), "value_kernel") != 0) {
            fprintf(stderr, "only one value_kernel launch with one 32-lane warp supported\n"); std::exit(2);
        }
        check(cudaDeviceSynchronize());
        std::memset(capture, 0, sizeof(Capture));
        int count;
        if (cuFuncGetAttribute(&count, CU_FUNC_ATTRIBUTE_NUM_REGS, p->f) != CUDA_SUCCESS ||
            count <= 0 || count > static_cast<int>(kRegisters)) std::exit(2);
        nregs = static_cast<unsigned>(count);
        const auto &instructions = nvbit_get_instrs(ctx, p->f);
        for (auto *instr : instructions) {
            const std::string sass = instr->getSass();
            if (instr->getOffset() == 0) {
                nvbit_insert_call(instr, "capture_constants", IPOINT_BEFORE);
                nvbit_add_call_arg_const_val64(instr, reinterpret_cast<uint64_t>(capture));
                for (int offset : {0x28, 0x118, 0x11c, 0x160, 0x164, 0x168, 0x16c, 0x170})
                    nvbit_add_call_arg_cbank_val(instr, 0, offset);
            }
            nvbit_insert_call(instr, "snapshot", IPOINT_BEFORE);
            nvbit_add_call_arg_const_val64(instr, reinterpret_cast<uint64_t>(capture));
            nvbit_add_call_arg_const_val32(instr, instr->getOffset());
            nvbit_add_call_arg_guard_pred_val(instr);
            nvbit_add_call_arg_pred_reg(instr);
            nvbit_add_call_arg_ureg_val(instr, 4);
            nvbit_add_call_arg_ureg_val(instr, 5);
            nvbit_add_call_arg_const_val32(instr, nregs);
            for (unsigned reg = 0; reg < nregs; ++reg) nvbit_add_call_arg_reg_val(instr, reg, true);
            bool is_load = sass.find("LDG.") != std::string::npos;
            bool is_store = sass.find("STG.") != std::string::npos;
            if (is_load || is_store) {
                int value_reg = -1;
                for (int operand = 0; operand < instr->getNumOperands(); ++operand) {
                    const auto *op = instr->getOperand(operand);
                    if (op->type != InstrType::OperandType::REG) continue;
                    if ((is_load && operand == 0) || (is_store && operand > 0)) {
                        value_reg = op->u.reg.num;
                    }
                }
                if (value_reg < 0) {
                    fprintf(stderr, "memory value register missing for %s\n", sass.c_str());
                    std::exit(2);
                }
                nvbit_insert_call(instr, "snapshot_memory", is_load ? IPOINT_AFTER : IPOINT_BEFORE);
                nvbit_add_call_arg_const_val64(instr, reinterpret_cast<uint64_t>(capture));
                nvbit_add_call_arg_const_val32(instr, instr->getOffset());
                nvbit_add_call_arg_guard_pred_val(instr);
                nvbit_add_call_arg_mref_addr64(instr);
                nvbit_add_call_arg_const_val32(instr, is_load ? 0u : 1u);
                nvbit_add_call_arg_reg_val(instr, value_reg);
            }
        }
        nvbit_enable_instrumented(ctx, p->f, true);
        launched = true;
    } else {
        check(cudaDeviceSynchronize());
        printf("CAPTURE {\"schema\":\"native_value_capture_v2\",\"launch_id\":0,\"function_name\":\"value_kernel\",\"function_unique_id\":null,\"grid\":[1,1,1],\"block\":[32,1,1],\"overflow\":%u,\"memory_overflow\":%u,\"nregs\":%u,\"constants\":[", capture->overflow, capture->memory_overflow, nregs);
        for (unsigned i=0;i<8;++i) printf("%s%u", i ? "," : "", capture->constants[i]);
        printf("],\"lanes\":[");
        for (unsigned lane=0;lane<kLanes;++lane) {
            printf("%s{\"lane\":%u,\"count\":%u,\"frames\":[",lane?",":"",lane,capture->count[lane]);
            for (unsigned i=0;i<capture->count[lane] && i<kFrames;++i) {
                auto &f=capture->frames[lane][i];
                printf("%s{\"pc\":%u,\"guard\":%u,\"active\":%u,\"predicates\":%u,\"ur4\":%u,\"ur5\":%u,\"regs\":[",i?",":"",f.pc,f.guard,f.active,f.predicates,f.ur4,f.ur5);
                for(unsigned r=0;r<nregs;++r) printf("%s%u",r?",":"",f.regs[r]);
                printf("]}");
            }
            printf("],\"memory\":[");
            for (unsigned i=0;i<capture->memory_count[lane] && i<kMemoryFrames;++i) {
                auto &m=capture->memory[lane][i];
                printf("%s{\"occurrence\":%u,\"pc\":%u,\"guard\":%u,\"address_u64\":%llu,\"address_u32\":%u,\"kind\":%u,\"value\":%u,\"cta\":[0,0,0],\"warp\":0,\"lane\":%u}", i?",":"",i,m.pc,m.guard,(unsigned long long)m.address,(unsigned)(m.address & 0xffffffffu),m.kind,m.value,lane);
            }
            printf("]}");
        }
        printf("]}\n");
        if (capture->overflow || capture->memory_overflow) std::exit(2);
    }
    internal = false;
}
void nvbit_at_ctx_term(CUcontext) {
    internal = true;
    if (capture) { check(cudaFree(capture)); capture = nullptr; }
    internal = false;
}
