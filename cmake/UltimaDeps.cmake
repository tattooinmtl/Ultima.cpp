# All third-party dependencies. Each is fetched via FetchContent, pinned to a
# specific tag, marked SYSTEM so their warnings do not pollute our build, and
# excluded from the ALL target so their own tests/examples do not build.

include(FetchContent)

# CMake 4.x removed compatibility with pre-3.5 cmake_minimum_required. Many
# widely-used deps still declare older minimums. This tells CMake to accept
# them under 3.5 policy behavior. Scoped to this file's FetchContent block.
set(CMAKE_POLICY_VERSION_MINIMUM 3.5 CACHE STRING "" FORCE)

# ---- doctest — unit test framework ------------------------------------------
FetchContent_Declare(doctest
    GIT_REPOSITORY https://github.com/doctest/doctest.git
    GIT_TAG        v2.4.11
    GIT_SHALLOW    ON
    SYSTEM
    EXCLUDE_FROM_ALL
)
FetchContent_MakeAvailable(doctest)

# Additional deps are fetched as their modules need them (Decision 04+).
# Keeping the initial dependency graph minimal to prove the build works first.
