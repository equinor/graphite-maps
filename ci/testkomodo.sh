# Invoked by komodo-releases/.github/workflows/run_tests_one_project.yml
run_tests () {
    pip install .[dev]
    pytest
}
