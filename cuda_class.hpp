#ifndef CUDA_CLASS_H
#define CUDA_CLASS_H 1

#include <iostream>
#include <string>
#include <vector>
#include <memory>
#include <cstdlib>

#include "./cuda_worker.hpp"

// // // CUDA (Compute Unified Device Architecture)
// // // is a parallel computing platform developed by NVIDIA.
class ComputeUnifiedDeviceArchitecture {
public:
    ComputeUnifiedDeviceArchitecture() = default;
    ~ComputeUnifiedDeviceArchitecture() = default;
    ComputeUnifiedDeviceArchitectureWorker* new_subscriber();
private:
    // std::vector<std::unique_ptr<ComputeUnifiedDeviceArchitectureWorker>> __subscribers__;
    std::vector<ComputeUnifiedDeviceArchitectureWorker*> __subscribers__;
};

#endif /* CUDA_CLASS_H */