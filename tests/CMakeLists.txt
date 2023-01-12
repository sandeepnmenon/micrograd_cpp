add_executable(variable_test variable_test.cpp)
target_link_libraries(variable_test micrograd_cpp)

add_executable(nn_test nn_test.cpp)
target_link_libraries(nn_test micrograd_cpp)
