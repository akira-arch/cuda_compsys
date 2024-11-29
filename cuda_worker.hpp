#ifndef CUDA_WORKER_H
#define CUDA_WORKER_H 1

#include <tuple>
#include <string>
#include <vector>
#include <memory>
#include <utility>
#include <cstddef>
#include <functional>

#include "./cuda_class.hpp"

class ComputeUnifiedDeviceArchitecture;

class ComputeUnifiedDeviceArchitectureWorker {
public:
    struct Dim {
        int grid;
        int block;
        int shared_mem = 0;
    };
    
    // ComputeUnifiedDeviceArchitectureWorker(ComputeUnifiedDeviceArchitecture*);
    ComputeUnifiedDeviceArchitectureWorker
        (const ComputeUnifiedDeviceArchitecture* _context) : _context(_context) {};
    
    ~ComputeUnifiedDeviceArchitectureWorker() = default;

    void get_info();
    // template<typename... T>
    //     void launch(std::function<void(T*...)>);
    // template<typename _fn>
    //     void launch(_fn);
    // template<typename _fn, typename... Args>// Args&& ...
    //     void launch_pack(_fn cb, std::tuple<Args...>);
    void size_init(size_t);
    void host_init(size_t);
    void device_init(size_t);
    void transform_idx(size_t, std::function<void(int *)>);
    void copy_host_to_device(size_t);
    void cleanup();
    int* get_device_copies_dx(size_t);
    int* get_host_copies_dx(size_t);
    void swap_idx(size_t);

    template<typename T>
        T set_dim(ComputeUnifiedDeviceArchitectureWorker::Dim*);

protected:
    const ComputeUnifiedDeviceArchitecture* _context; // Pointer to device architecture context
    // Pair of integer pointers for host and device copies, along with size
    // pair<int*>{host copies, device copies}, size
    std::vector<std::pair<int*, int*>, std::allocator<std::pair<int*, int*>>> _copies;
    // Size of data being copied
    std::size_t _size = 512 * 4; // Default size of 512 units
    // Pointer to dimension information
    ComputeUnifiedDeviceArchitectureWorker::Dim* _dim; // Dimension data for copying

    template<typename T, std::size_t... Is>
        auto vec_to_tuple_impl(const std::vector<T>& vec, std::index_sequence<Is...>) {
            return std::make_tuple(vec[Is]...);
        };

    template<typename T, std::size_t N>
        auto vec_to_tuple(const std::vector<T>& vec) {
            return vec_to_tuple_impl(vec, std::make_index_sequence<N>{});
        };

    template<typename _fn, typename... Args>
        void laucnh_pack(_fn cb, std::tuple<Args...> args);
};

#endif /* CUDA_WORKER_H */