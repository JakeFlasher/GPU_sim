from pathlib import Path


_TRACER_SOURCE = (
    Path(__file__).resolve().parent / "tracer_tool" / "tracer_tool.cu"
)


def test_static_shared_memory_attribute_fails_closed_before_serialization() -> None:
    source = _TRACER_SOURCE.read_text(encoding="utf-8")
    attribute = "CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES"
    assert source.count(attribute) == 1

    query = source.index(attribute)
    status_check = source.index(
        "shmem_static_status != CUDA_SUCCESS", query
    )
    fatal = source.index(
        "static_shared_memory_attribute_unavailable", status_check
    )
    process_exit = source.index("_exit(EXIT_FAILURE)", fatal)
    combined = source.index("const uint64_t shmem_combined_nbytes", process_exit)
    static_setter = source.index(
        "set_shared_memory_static_bytes", combined
    )
    dynamic_setter = source.index(
        "set_shared_memory_dynamic_bytes", static_setter
    )

    query_line_start = source.rfind("\n", 0, query)
    query_context = source[source.rfind("\n", 0, query_line_start) : query]
    assert "CUDA_SAFECALL" not in query_context
    assert "int shmem_static_nbytes = -1;" in source[:query]
    assert query < status_check < fatal < process_exit < combined
    assert combined < static_setter < dynamic_setter


def test_register_attribute_fails_closed_before_serialization() -> None:
    source = _TRACER_SOURCE.read_text(encoding="utf-8")
    query = source.index("CU_FUNC_ATTRIBUTE_NUM_REGS")
    status_check = source.index("nregs_status != CUDA_SUCCESS", query)
    fatal = source.index("register_attribute_unavailable", status_check)
    process_exit = source.index("_exit(EXIT_FAILURE)", fatal)
    setter = source.index("set_number_of_registers", process_exit)

    assert "int nregs = -1;" in source[:query]
    assert query < status_check < fatal < process_exit < setter


def test_tool_init_suppresses_callbacks_during_cuda_setup() -> None:
    source = _TRACER_SOURCE.read_text(encoding="utf-8")
    function_start = source.index("void nvbit_tool_init(CUcontext ctx)")
    function_end = source.index("void nvbit_at_ctx_term(CUcontext ctx)", function_start)
    body = source[function_start:function_end]

    callback_guard_on = body.index("skip_callback_flag = true;")
    context_setup = body.index("init_context_state(ctx);")
    architecture_query = body.index("resolve_binary_version_for_current_device()")
    callback_guard_off = body.index("skip_callback_flag = false;")
    mutex_unlock = body.index("pthread_mutex_unlock(&mutex);")

    assert body.count("skip_callback_flag = true;") == 1
    assert body.count("skip_callback_flag = false;") == 1
    assert callback_guard_on < context_setup < architecture_query
    assert architecture_query < callback_guard_off < mutex_unlock


def test_context_init_establishes_identity_before_application_callbacks() -> None:
    source = _TRACER_SOURCE.read_text(encoding="utf-8")
    function_start = source.index("void nvbit_at_ctx_init(CUcontext ctx)")
    function_end = source.index("void init_context_state(CUcontext ctx)", function_start)
    body = source[function_start:function_end]

    state_publication = body.index("ctx_state_map[ctx] = ctx_state;")
    callback_guard_on = body.index("skip_callback_flag = true;")
    device_count = body.index("initialize_or_validate_device_count(")
    context_device = body.index("checked_current_context_device_id(")
    callback_guard_off = body.index("skip_callback_flag = false;")
    mutex_unlock = body.index("pthread_mutex_unlock(&mutex);")

    assert body.count("skip_callback_flag = true;") == 1
    assert body.count("skip_callback_flag = false;") == 1
    assert state_publication < callback_guard_on < device_count < context_device
    assert context_device < callback_guard_off < mutex_unlock
    assert "cudaMalloc" not in body
    assert "cudaGetDeviceCount" not in body
