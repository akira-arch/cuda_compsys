/* https://www.nvidia.com/docs/IO/116711/sc11-cuda-c-basics.pdf */

#include "./cuda_class.hpp"
#include "cuda_worker.hpp"

 // Get information about the worker
void ComputeUnifiedDeviceArchitectureWorker::get_info() {
    std::cout << "Sizee: " << this->_size << std::endl;
    std::cout << "<text>: " << this->_copies.size() << std::endl;
    std::cout << "GPU Blocks: " << this->_dim->grid << std::endl;
};

// Create a new subscriber (worker)
ComputeUnifiedDeviceArchitectureWorker* ComputeUnifiedDeviceArchitecture::new_subscriber() {
    ComputeUnifiedDeviceArchitectureWorker* worker = new ComputeUnifiedDeviceArchitectureWorker(this);
    __subscribers__.push_back(worker);

    // return new ComputeUnifiedDeviceArchitectureWorker(this);
    return worker;
};

void ComputeUnifiedDeviceArchitectureWorker::size_init(size_t quantity) {
    this->_size = quantity * sizeof(int);
};

// Set dimension information
template<>
void ComputeUnifiedDeviceArchitectureWorker::set_dim<void>(ComputeUnifiedDeviceArchitectureWorker::Dim* _dim) {
    this->_dim = _dim;
};

template<>
const ComputeUnifiedDeviceArchitectureWorker::Dim* ComputeUnifiedDeviceArchitectureWorker::set_dim<const ComputeUnifiedDeviceArchitectureWorker::Dim*>(ComputeUnifiedDeviceArchitectureWorker::Dim* _dim) {
    this->_dim = _dim;
    return this->_dim;
};

void ComputeUnifiedDeviceArchitectureWorker::host_init(size_t quantity) {
    for (size_t idx = 0; idx < quantity; idx++) {
        // this->_copies.emplace_back((int *)std::malloc(this->_size), this->_copies[idx].second);
        this->_copies.emplace_back((int *)std::malloc(this->_size), nullptr);
    }
};

void ComputeUnifiedDeviceArchitectureWorker::device_init(size_t quantity) {
    for (size_t idx = 0; idx < quantity; idx++) {
        cudaMalloc((void **)&this->_copies[idx].second, this->_size);
    }
};

// Transform data using a callback function
void ComputeUnifiedDeviceArchitectureWorker::transform_idx(size_t idx, std::function<void(int *)> cb) {
    cb(this->_copies[idx].first);
};

void ComputeUnifiedDeviceArchitectureWorker::copy_host_to_device(size_t quantity) {
    for (size_t idx = 0; idx < quantity; idx++) {
        cudaMemcpy(this->_copies[idx].second, this->_copies[idx].first, this->_size, cudaMemcpyHostToDevice);
    }
};

int* ComputeUnifiedDeviceArchitectureWorker::get_host_copies_dx(size_t idx) {
    return this->_copies[idx].first;
};

int* ComputeUnifiedDeviceArchitectureWorker::get_device_copies_dx(size_t idx) {
    return this->_copies[idx].second;
};

// Swap device and host data
void ComputeUnifiedDeviceArchitectureWorker::swap_idx(size_t idx) {
    cudaMemcpy(this->_copies[idx].first, this->_copies[idx].second, this->_size, cudaMemcpyDeviceToHost);
};

// Clean up resources
void ComputeUnifiedDeviceArchitectureWorker::cleanup() {
    for (auto subscriber: this->_copies) {
        free(subscriber.first);
        cudaFree(subscriber.second);
    }
};

// CUDA kernel for element-wise addition
__global__ void add(int *a, int *b, int *c) {
    c[blockIdx.x] = a[blockIdx.x] + b[blockIdx.x];
}

#define N 512

// First version: Random integers without a minimum
void random_ints(int* a, int n, int max) {
    // Seed the random number generator
    static bool seeded = false;
    if (!seeded) {
        srand(static_cast<unsigned int>(time(nullptr)));
        seeded = true;
    }

    // Generate and store N random integers
    for (int i = 0; i < n; ++i) {
        a[i] = rand() % (max + 1);
    }
}

// Second version: Random integers with a minimum
void random_ints(int* a, int n, int max, int min) {
    // Seed the random number generator
    static bool seeded = false;
    if (!seeded) {
        srand(static_cast<unsigned int>(time(nullptr)));
        seeded = false;  // Reset the seed flag
    }

    // Generate and store N random integers within the specified range
    for (int i = 0; i < n; ++i) {
        a[i] = rand() % ((max - min + 1)) + min;
    }
}


int main (void) {
    ComputeUnifiedDeviceArchitecture constructor;
    ComputeUnifiedDeviceArchitectureWorker* subscriber = constructor.new_subscriber();
    subscriber->size_init(N);
    subscriber->device_init(3);
    subscriber->host_init(3);

    // Transform data using random number generation
    subscriber->transform_idx(0, [](int * _it) -> void { random_ints(_it, N, 100, 0); });
    subscriber->transform_idx(1, [](int * _it) -> void { random_ints(_it, N, 100, 0); });
    subscriber->copy_host_to_device(2);

    ComputeUnifiedDeviceArchitectureWorker::Dim* dim = new ComputeUnifiedDeviceArchitectureWorker::Dim { 
        .grid = N, .block = 1
    };
    subscriber->set_dim<void>(dim);

    // Launch add() kernel on GPU with N blocks
    add<<<N, 1>>>(
        subscriber->get_device_copies_dx(0),
        subscriber->get_device_copies_dx(1),
        subscriber->get_device_copies_dx(2)
    );

    // Copy result back to host
    subscriber->swap_idx(2);
    subscriber->cleanup();
    subscriber->get_info();
    return 0;
}
