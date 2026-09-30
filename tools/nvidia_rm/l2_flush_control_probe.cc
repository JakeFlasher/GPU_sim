// Probe the public NV2080 FB full-cache writeback/invalidate control.
// This issues no GPU kernel and is not a test or verification workflow.

#include <fcntl.h>
#include <cerrno>
#include <cstdio>
#include <cstring>
#include <initializer_list>
#include <sys/ioctl.h>
#include <unistd.h>

#define NV_PLATFORM_MAX_IOCTL_SIZE 16384
#include "nv.h"
#include "nvtypes.h"
#include "nvos.h"
#include "nv_escape.h"
#include "nv-unix-nvos-params-wrappers.h"
#include <class/cl0080.h>
#include <class/cl2080.h>
#include <ctrl/ctrl2080/ctrl2080fb.h>

static NvHandle alloc_object(int fd, NvHandle root, NvHandle parent, NvU32 cls,
                             void *params, NvU32 size) {
  NVOS21_PARAMETERS p = {};
  p.hRoot = root; p.hObjectParent = parent; p.hClass = cls;
  p.pAllocParms = params; p.paramsSize = size;
  if (ioctl(fd, __NV_IOWR(NV_ESC_RM_ALLOC, p), &p) != 0) {
    std::perror("RM_ALLOC"); return 0;
  }
  std::fprintf(stderr, "alloc class=0x%x status=0x%x handle=0x%x\n",
               cls, p.status, p.hObjectNew);
  return p.status == 0 ? p.hObjectNew : 0;
}

static NvU32 control(int fd, NvHandle client, NvHandle object, NvU32 cmd,
                     void *params, NvU32 size) {
  NVOS54_PARAMETERS p = {};
  p.hClient = client; p.hObject = object; p.cmd = cmd;
  p.params = params; p.paramsSize = size;
  int ret = ioctl(fd, __NV_IOWR(NV_ESC_RM_CONTROL, p), &p);
  std::fprintf(stderr, "control cmd=0x%08x ret=%d errno=%d status=0x%08x\n",
               cmd, ret, ret ? errno : 0, p.status);
  return p.status;
}

static void free_object(int fd, NvHandle root, NvHandle parent, NvHandle obj) {
  NVOS00_PARAMETERS p = {};
  p.hRoot = root; p.hObjectParent = parent; p.hObjectOld = obj;
  ioctl(fd, __NV_IOWR(NV_ESC_RM_FREE, p), &p);
}

int main() {
  int ctl = open("/dev/nvidiactl", O_RDWR | O_CLOEXEC);
  int dev = open("/dev/nvidia0", O_RDWR | O_CLOEXEC);
  if (ctl < 0 || dev < 0) { std::perror("open"); return 2; }
  if (ioctl(dev, __NV_IOWR(NV_ESC_REGISTER_FD, ctl), &ctl) != 0) {
    std::perror("NV_ESC_REGISTER_FD"); return 3;
  }

  NvHandle root = alloc_object(ctl, 0, 0, NV01_ROOT_CLIENT, nullptr, 0);
  NV0080_ALLOC_PARAMETERS ap = {};
  ap.deviceId = 0;
  NvHandle device = alloc_object(ctl, root, root, NV01_DEVICE_0, &ap, sizeof(ap));
  NV2080_ALLOC_PARAMETERS sp = {};
  NvHandle sub = alloc_object(ctl, root, device, NV20_SUBDEVICE_0, &sp, sizeof(sp));
  if (!root || !device || !sub) return 4;

  for (NvU32 flags : {0x38u, 0x3bu}) {
    NV2080_CTRL_FB_FLUSH_GPU_CACHE_PARAMS params = {};
    params.addressAlign = 1;
    params.flags = flags;
    control(ctl, root, sub, NV2080_CTRL_CMD_FB_FLUSH_GPU_CACHE,
            &params, sizeof(params));
  }

  free_object(ctl, root, device, sub);
  free_object(ctl, root, root, device);
  free_object(ctl, root, root, root);
  close(dev); close(ctl);
  return 0;
}
