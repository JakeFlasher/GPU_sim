MKDIR_P = mkdir -p

ODIR=obj
SRCDIR=src

ifeq ($(DEBUG),1)
	CXXFLAGS = -Wall -O0 -g3 -fPIC -std=c++20
else ifeq ($(DEBUG_TOOL),1)
	CXXFLAGS = -Wall -O0 -g3 -fPIC -std=c++20
else
	CXXFLAGS = -Wall -O3 -g3 -fPIC -std=c++20
endif

TRACES_ENHANCED_PROTO_DIR=dynamic_trace
TRACES_ENHANCED_DIR=.
TRACES_ENHANCED_OBJ=$(wildcard $(TRACES_ENHANCED_DIR)/obj/*.o)

# Protocol Buffer setup
PROTOC?=protoc
PROTO_FILES=$(wildcard $(TRACES_ENHANCED_PROTO_DIR)/*.proto)

# Define directories for generated files
PROTO_GEN_DIR=$(TRACES_ENHANCED_DIR)/pb_trace
PROTO_SRC_DIR=$(PROTO_GEN_DIR)/src
PROTO_HDR_DIR=$(PROTO_GEN_DIR)/include
PROTO_OBJ_DIR=$(PROTO_GEN_DIR)/obj

# Resolve the full protobuf include/link contract through pkg-config. The
# pinned prefix is checkout-local, while its .pc files retain the path from the
# checkout where the environment was created; --define-prefix relocates those
# paths from the .pc file's current directory.
PKG_CONFIG?=pkg-config
PKG_CONFIG_FLAGS?=--define-prefix
PROTOBUF_INCLUDES:=$(shell $(PKG_CONFIG) $(PKG_CONFIG_FLAGS) --cflags protobuf 2>/dev/null)
PROTOBUF_LIBS:=$(shell $(PKG_CONFIG) $(PKG_CONFIG_FLAGS) --libs protobuf 2>/dev/null)
ifeq ($(PROTOBUF_INCLUDES),)
$(error pkg-config could not resolve protobuf compile flags; ensure PKG_CONFIG_PATH names the pinned protobuf pkgconfig directory)
endif
ifeq ($(PROTOBUF_LIBS),)
$(error pkg-config could not resolve protobuf link flags; ensure PKG_CONFIG_PATH names the pinned protobuf pkgconfig directory)
endif

# Define path transformations for proto files
PROTO_SRCS=$(patsubst $(TRACES_ENHANCED_PROTO_DIR)/%.proto,$(PROTO_SRC_DIR)/%.pb.cc,$(PROTO_FILES))
PROTO_HDRS=$(patsubst $(TRACES_ENHANCED_PROTO_DIR)/%.proto,$(PROTO_HDR_DIR)/%.pb.h,$(PROTO_FILES))
PROTO_OBJS=$(patsubst $(PROTO_SRC_DIR)/%.pb.cc,$(PROTO_OBJ_DIR)/%.pb.o,$(PROTO_SRCS))

# Private generated-code include path. A generic `INCLUDES` variable is unsafe
# here because command-line variables are inherited by recursive make and
# override ordinary Makefile appends.
TRACE_PROTO_INCLUDES=$(PROTOBUF_INCLUDES) -I$(TRACES_ENHANCED_DIR) -I$(PROTO_HDR_DIR)


CFLAGS=$(CXXFLAGS) -I$(CUDA_INSTALL_PATH)/include -fPIC

# NOTE: this Makefile only compiles .o files and does not link a final binary,
# so LIBS is only consumed if a downstream rule (trace_printer, tracer_tool.so)
# reuses it. The tracer_tool/Makefile link rules define their own PROTOBUF_LIBS
# via pkg-config; LIBS here exists only to keep the cc rule's environment
# stable. Assign (not +=) so we don't inherit dangling state from callers.
LIBS=-lz -pthread $(PROTOBUF_LIBS)

SOURCES := $(shell find $(SRCDIR) -name '*.cc')
OBJ := $(patsubst $(SRCDIR)/%,$(ODIR)/%,$(SOURCES:.cc=.o))

.PHONY: traces_enhanced directories protobuf clean depend

traces_enhanced: directories protobuf depend

directories: ${ODIR} $(PROTO_SRC_DIR) $(PROTO_HDR_DIR) $(PROTO_OBJ_DIR)

${ODIR} $(PROTO_SRC_DIR) $(PROTO_HDR_DIR) $(PROTO_OBJ_DIR):
	${MKDIR_P} $@

depend: $(OBJ)

$(ODIR)/%.o: $(SRCDIR)/%.cc | ${ODIR}
	$(CXX) -c -o $@ $< $(CFLAGS)

# Protocol buffer compilation rules
$(PROTO_SRC_DIR)/%.pb.cc $(PROTO_HDR_DIR)/%.pb.h: $(TRACES_ENHANCED_PROTO_DIR)/%.proto | $(PROTO_SRC_DIR) $(PROTO_HDR_DIR)
	$(PROTOC) --cpp_out=$(PROTO_SRC_DIR) --proto_path=$(TRACES_ENHANCED_PROTO_DIR) $<
	@mv $(PROTO_SRC_DIR)/$*.pb.h $(PROTO_HDR_DIR)/$*.pb.h

$(PROTO_OBJ_DIR)/%.pb.o: $(PROTO_SRC_DIR)/%.pb.cc $(PROTO_HDR_DIR)/%.pb.h | $(PROTO_OBJ_DIR)
	$(CXX) -c $(TRACE_PROTO_INCLUDES) -fPIC -o $@ $<


# Generate Protocol Buffer files
protobuf: $(PROTO_SRCS) $(PROTO_HDRS) $(PROTO_OBJS)
	@echo "Protocol buffer files generated"
# MOD. End. ENHANCED TRACES

clean:
	rm -rf $(ODIR)
	rm -rf $(PROTO_OBJ_DIR)
