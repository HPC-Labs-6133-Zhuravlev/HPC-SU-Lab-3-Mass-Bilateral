#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t err__ = (call);                                             \
        if (err__ != cudaSuccess) {                                             \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__        \
                      << " -> " << cudaGetErrorString(err__) << std::endl;      \
            std::exit(EXIT_FAILURE);                                            \
        }                                                                       \
    } while (0)

#pragma pack(push, 1)
struct BMPFileHeader {
    uint16_t bfType;
    uint32_t bfSize;
    uint16_t bfReserved1;
    uint16_t bfReserved2;
    uint32_t bfOffBits;
};

struct BMPInfoHeader {
    uint32_t biSize;
    int32_t  biWidth;
    int32_t  biHeight;
    uint16_t biPlanes;
    uint16_t biBitCount;
    uint32_t biCompression;
    uint32_t biSizeImage;
    int32_t  biXPelsPerMeter;
    int32_t  biYPelsPerMeter;
    uint32_t biClrUsed;
    uint32_t biClrImportant;
};
#pragma pack(pop)

struct GpuTimingStats {
    float kernelAvgMs = 0.0f;
    float kernelMinMs = 0.0f;
    float kernelMaxMs = 0.0f;
    float kernelStdMs = 0.0f;
    double setupAndCopyToDeviceMs = 0.0;
    double copyToHostMs = 0.0;
    double estimatedTotalOneRunMs = 0.0;
};

static unsigned char clampToByte(float value) {
    if (value < 0.0f) return 0;
    if (value > 255.0f) return 255;
    return static_cast<unsigned char>(value + 0.5f);
}

bool readBMPGray(const std::string& filename,
                 std::vector<unsigned char>& data,
                 int& width,
                 int& height) {
    std::ifstream in(filename, std::ios::binary);
    if (!in) {
        std::cerr << "Cannot open input BMP file: " << filename << std::endl;
        return false;
    }

    BMPFileHeader fileHeader{};
    BMPInfoHeader infoHeader{};

    in.read(reinterpret_cast<char*>(&fileHeader), sizeof(fileHeader));
    in.read(reinterpret_cast<char*>(&infoHeader), sizeof(infoHeader));

    if (!in) {
        std::cerr << "Cannot read BMP headers." << std::endl;
        return false;
    }

    if (fileHeader.bfType != 0x4D42) {
        std::cerr << "Input file is not BMP." << std::endl;
        return false;
    }

    if (infoHeader.biSize < 40) {
        std::cerr << "Unsupported BMP header size." << std::endl;
        return false;
    }

    if (infoHeader.biCompression != 0) {
        std::cerr << "Only uncompressed BMP is supported." << std::endl;
        return false;
    }

    if (!(infoHeader.biBitCount == 8 ||
          infoHeader.biBitCount == 24 ||
          infoHeader.biBitCount == 32)) {
        std::cerr << "Only 8-bit, 24-bit and 32-bit BMP are supported." << std::endl;
        return false;
    }

    if (infoHeader.biWidth <= 0 || infoHeader.biHeight == 0) {
        std::cerr << "Invalid BMP dimensions." << std::endl;
        return false;
    }

    width = infoHeader.biWidth;
    height = infoHeader.biHeight > 0 ? infoHeader.biHeight : -infoHeader.biHeight;
    const bool topDown = infoHeader.biHeight < 0;

    std::vector<unsigned char> paletteGray(256);
    for (int i = 0; i < 256; ++i) {
        paletteGray[i] = static_cast<unsigned char>(i);
    }

    if (infoHeader.biBitCount == 8) {
        uint32_t colors = infoHeader.biClrUsed ? infoHeader.biClrUsed : 256;
        colors = std::min(colors, 256u);

        in.seekg(sizeof(BMPFileHeader) + infoHeader.biSize, std::ios::beg);

        for (uint32_t i = 0; i < colors; ++i) {
            unsigned char bgra[4]{};
            in.read(reinterpret_cast<char*>(bgra), 4);

            const unsigned char b = bgra[0];
            const unsigned char g = bgra[1];
            const unsigned char r = bgra[2];

            paletteGray[i] = static_cast<unsigned char>(
                0.299f * r + 0.587f * g + 0.114f * b + 0.5f
            );
        }
    }

    const int bytesPerPixel = infoHeader.biBitCount / 8;
    const int rowStride = ((width * bytesPerPixel + 3) / 4) * 4;

    data.assign(static_cast<size_t>(width) * height, 0);

    std::vector<unsigned char> row(rowStride);

    in.seekg(fileHeader.bfOffBits, std::ios::beg);

    for (int fileRow = 0; fileRow < height; ++fileRow) {
        in.read(reinterpret_cast<char*>(row.data()), rowStride);

        if (!in) {
            std::cerr << "Cannot read BMP pixel data." << std::endl;
            return false;
        }

        const int y = topDown ? fileRow : (height - 1 - fileRow);

        for (int x = 0; x < width; ++x) {
            unsigned char grayValue = 0;

            if (infoHeader.biBitCount == 8) {
                grayValue = paletteGray[row[x]];
            } else if (infoHeader.biBitCount == 24) {
                const int p = x * 3;
                const unsigned char b = row[p + 0];
                const unsigned char g = row[p + 1];
                const unsigned char r = row[p + 2];

                grayValue = static_cast<unsigned char>(
                    0.299f * r + 0.587f * g + 0.114f * b + 0.5f
                );
            } else {
                const int p = x * 4;
                const unsigned char b = row[p + 0];
                const unsigned char g = row[p + 1];
                const unsigned char r = row[p + 2];

                grayValue = static_cast<unsigned char>(
                    0.299f * r + 0.587f * g + 0.114f * b + 0.5f
                );
            }

            data[static_cast<size_t>(y) * width + x] = grayValue;
        }
    }

    return true;
}

bool writeBMPGray24(const std::string& filename,
                    const std::vector<unsigned char>& data,
                    int width,
                    int height) {
    if (width <= 0 || height <= 0 ||
        data.size() != static_cast<size_t>(width) * height) {
        std::cerr << "Invalid data for BMP writing." << std::endl;
        return false;
    }

    const int bytesPerPixel = 3;
    const int rowStride = ((width * bytesPerPixel + 3) / 4) * 4;
    const uint32_t imageSize = static_cast<uint32_t>(rowStride * height);

    BMPFileHeader fileHeader{};
    BMPInfoHeader infoHeader{};

    fileHeader.bfType = 0x4D42;
    fileHeader.bfOffBits = sizeof(BMPFileHeader) + sizeof(BMPInfoHeader);
    fileHeader.bfSize = fileHeader.bfOffBits + imageSize;

    infoHeader.biSize = sizeof(BMPInfoHeader);
    infoHeader.biWidth = width;
    infoHeader.biHeight = height;
    infoHeader.biPlanes = 1;
    infoHeader.biBitCount = 24;
    infoHeader.biCompression = 0;
    infoHeader.biSizeImage = imageSize;

    std::ofstream out(filename, std::ios::binary);
    if (!out) {
        std::cerr << "Cannot create output BMP file: " << filename << std::endl;
        return false;
    }

    out.write(reinterpret_cast<const char*>(&fileHeader), sizeof(fileHeader));
    out.write(reinterpret_cast<const char*>(&infoHeader), sizeof(infoHeader));

    std::vector<unsigned char> row(rowStride, 0);

    for (int y = height - 1; y >= 0; --y) {
        std::fill(row.begin(), row.end(), 0);

        for (int x = 0; x < width; ++x) {
            const unsigned char v = data[static_cast<size_t>(y) * width + x];

            row[x * 3 + 0] = v;
            row[x * 3 + 1] = v;
            row[x * 3 + 2] = v;
        }

        out.write(reinterpret_cast<const char*>(row.data()), rowStride);
    }

    return true;
}

void bilateralCPU(const std::vector<unsigned char>& input,
                  std::vector<unsigned char>& output,
                  int width,
                  int height,
                  float sigmaD,
                  float sigmaR) {
    output.resize(static_cast<size_t>(width) * height);

    const float inv2SigmaD2 = 1.0f / (2.0f * sigmaD * sigmaD);
    const float inv2SigmaR2 = 1.0f / (2.0f * sigmaR * sigmaR);

    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            const float center = static_cast<float>(
                input[static_cast<size_t>(y) * width + x]
            );

            float weightedSum = 0.0f;
            float weightNorm = 0.0f;

            for (int dy = -1; dy <= 1; ++dy) {
                for (int dx = -1; dx <= 1; ++dx) {
                    int xx = x + dx;
                    int yy = y + dy;

                    if (xx < 0) xx = 0;
                    if (yy < 0) yy = 0;
                    if (xx >= width) xx = width - 1;
                    if (yy >= height) yy = height - 1;

                    const float value = static_cast<float>(
                        input[static_cast<size_t>(yy) * width + xx]
                    );

                    const float spatialWeight = std::exp(
                        -static_cast<float>(dx * dx + dy * dy) * inv2SigmaD2
                    );

                    const float diff = value - center;

                    const float rangeWeight = std::exp(
                        -(diff * diff) * inv2SigmaR2
                    );

                    const float weight = spatialWeight * rangeWeight;

                    weightedSum += weight * value;
                    weightNorm += weight;
                }
            }

            output[static_cast<size_t>(y) * width + x] =
                clampToByte(weightedSum / weightNorm);
        }
    }
}

__device__ __forceinline__ unsigned char clampToByteDevice(float value) {
    if (value < 0.0f) return 0;
    if (value > 255.0f) return 255;
    return static_cast<unsigned char>(value + 0.5f);
}

__global__ void bilateralKernel(cudaTextureObject_t texObj,
                                unsigned char* output,
                                int width,
                                int height,
                                float sigmaD,
                                float sigmaR) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) {
        return;
    }

    const float inv2SigmaD2 = 1.0f / (2.0f * sigmaD * sigmaD);
    const float inv2SigmaR2 = 1.0f / (2.0f * sigmaR * sigmaR);

    const float center = static_cast<float>(
        tex2D<unsigned char>(texObj, x + 0.5f, y + 0.5f)
    );

    float weightedSum = 0.0f;
    float weightNorm = 0.0f;

    // 9-point mask: 3 x 3 neighborhood.
    for (int dy = -1; dy <= 1; ++dy) {
        for (int dx = -1; dx <= 1; ++dx) {
            int xx = x + dx;
            int yy = y + dy;

            // Missing values for edge rows and columns are taken from nearest pixels.
            if (xx < 0) xx = 0;
            if (yy < 0) yy = 0;
            if (xx >= width) xx = width - 1;
            if (yy >= height) yy = height - 1;

            const float value = static_cast<float>(
                tex2D<unsigned char>(texObj, xx + 0.5f, yy + 0.5f)
            );

            const float spatialWeight = expf(
                -static_cast<float>(dx * dx + dy * dy) * inv2SigmaD2
            );

            const float diff = value - center;

            const float rangeWeight = expf(
                -(diff * diff) * inv2SigmaR2
            );

            const float weight = spatialWeight * rangeWeight;

            weightedSum += weight * value;
            weightNorm += weight;
        }
    }

    output[static_cast<size_t>(y) * width + x] =
        clampToByteDevice(weightedSum / weightNorm);
}

static float meanValue(const std::vector<float>& values) {
    if (values.empty()) return 0.0f;
    const float sum = std::accumulate(values.begin(), values.end(), 0.0f);
    return sum / static_cast<float>(values.size());
}

static float stdValue(const std::vector<float>& values, float mean) {
    if (values.size() <= 1) return 0.0f;

    float sum = 0.0f;

    for (float v : values) {
        const float d = v - mean;
        sum += d * d;
    }

    return std::sqrt(sum / static_cast<float>(values.size() - 1));
}

void bilateralGPU(const std::vector<unsigned char>& input,
                  std::vector<unsigned char>& output,
                  int width,
                  int height,
                  float sigmaD,
                  float sigmaR,
                  int measuredRuns,
                  int warmupRuns,
                  GpuTimingStats& stats) {
    output.resize(static_cast<size_t>(width) * height);

    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<unsigned char>();

    cudaArray_t cudaArrayInput = nullptr;
    cudaTextureObject_t texObj = 0;
    unsigned char* dOutput = nullptr;

    const size_t imageBytes =
        static_cast<size_t>(width) * height * sizeof(unsigned char);

    const auto setupStart = std::chrono::high_resolution_clock::now();

    CUDA_CHECK(cudaMallocArray(
        &cudaArrayInput,
        &channelDesc,
        width,
        height
    ));

    CUDA_CHECK(cudaMemcpy2DToArray(
        cudaArrayInput,
        0,
        0,
        input.data(),
        static_cast<size_t>(width) * sizeof(unsigned char),
        static_cast<size_t>(width) * sizeof(unsigned char),
        height,
        cudaMemcpyHostToDevice
    ));

    cudaResourceDesc resourceDesc{};
    resourceDesc.resType = cudaResourceTypeArray;
    resourceDesc.res.array.array = cudaArrayInput;

    cudaTextureDesc textureDesc{};
    textureDesc.addressMode[0] = cudaAddressModeClamp;
    textureDesc.addressMode[1] = cudaAddressModeClamp;
    textureDesc.filterMode = cudaFilterModePoint;
    textureDesc.readMode = cudaReadModeElementType;
    textureDesc.normalizedCoords = 0;

    CUDA_CHECK(cudaCreateTextureObject(
        &texObj,
        &resourceDesc,
        &textureDesc,
        nullptr
    ));

    CUDA_CHECK(cudaMalloc(&dOutput, imageBytes));
    CUDA_CHECK(cudaDeviceSynchronize());

    const auto setupStop = std::chrono::high_resolution_clock::now();

    stats.setupAndCopyToDeviceMs =
        std::chrono::duration<double, std::milli>(setupStop - setupStart).count();

    const dim3 block(16, 16);
    const dim3 grid(
        (width + block.x - 1) / block.x,
        (height + block.y - 1) / block.y
    );

    for (int i = 0; i < warmupRuns; ++i) {
        bilateralKernel<<<grid, block>>>(
            texObj,
            dOutput,
            width,
            height,
            sigmaD,
            sigmaR
        );

        CUDA_CHECK(cudaGetLastError());
    }

    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<float> kernelTimes;
    kernelTimes.reserve(measuredRuns);

    cudaEvent_t startEvent;
    cudaEvent_t stopEvent;

    CUDA_CHECK(cudaEventCreate(&startEvent));
    CUDA_CHECK(cudaEventCreate(&stopEvent));

    for (int i = 0; i < measuredRuns; ++i) {
        CUDA_CHECK(cudaEventRecord(startEvent));

        bilateralKernel<<<grid, block>>>(
            texObj,
            dOutput,
            width,
            height,
            sigmaD,
            sigmaR
        );

        CUDA_CHECK(cudaGetLastError());

        CUDA_CHECK(cudaEventRecord(stopEvent));
        CUDA_CHECK(cudaEventSynchronize(stopEvent));

        float elapsedMs = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&elapsedMs, startEvent, stopEvent));

        kernelTimes.push_back(elapsedMs);
    }

    CUDA_CHECK(cudaEventDestroy(startEvent));
    CUDA_CHECK(cudaEventDestroy(stopEvent));

    stats.kernelAvgMs = meanValue(kernelTimes);
    stats.kernelMinMs = *std::min_element(kernelTimes.begin(), kernelTimes.end());
    stats.kernelMaxMs = *std::max_element(kernelTimes.begin(), kernelTimes.end());
    stats.kernelStdMs = stdValue(kernelTimes, stats.kernelAvgMs);

    const auto copyStart = std::chrono::high_resolution_clock::now();

    CUDA_CHECK(cudaMemcpy(
        output.data(),
        dOutput,
        imageBytes,
        cudaMemcpyDeviceToHost
    ));

    const auto copyStop = std::chrono::high_resolution_clock::now();

    stats.copyToHostMs =
        std::chrono::duration<double, std::milli>(copyStop - copyStart).count();

    stats.estimatedTotalOneRunMs =
        stats.setupAndCopyToDeviceMs + stats.kernelAvgMs + stats.copyToHostMs;

    CUDA_CHECK(cudaDestroyTextureObject(texObj));
    CUDA_CHECK(cudaFree(dOutput));
    CUDA_CHECK(cudaFreeArray(cudaArrayInput));
}

void compareImages(const std::vector<unsigned char>& a,
                   const std::vector<unsigned char>& b,
                   int& maxAbsDiff,
                   double& meanAbsDiff) {
    maxAbsDiff = 0;
    long long sumAbsDiff = 0;

    for (size_t i = 0; i < a.size(); ++i) {
        const int diff = std::abs(static_cast<int>(a[i]) - static_cast<int>(b[i]));
        maxAbsDiff = std::max(maxAbsDiff, diff);
        sumAbsDiff += diff;
    }

    meanAbsDiff =
        static_cast<double>(sumAbsDiff) / static_cast<double>(a.size());
}

int main(int argc, char** argv) {
    std::string inputFile = "input.bmp";
    float sigmaD = 1.0f;
    float sigmaR = 30.0f;
    int measuredGpuRuns = 20;
    int warmupGpuRuns = 3;

    if (argc >= 2) {
        inputFile = argv[1];
    }

    if (argc >= 3) {
        sigmaD = std::stof(argv[2]);
    }

    if (argc >= 4) {
        sigmaR = std::stof(argv[3]);
    }

    if (argc >= 5) {
        measuredGpuRuns = std::stoi(argv[4]);
    }

    if (argc >= 6) {
        warmupGpuRuns = std::stoi(argv[5]);
    }

    if (sigmaD <= 0.0f || sigmaR <= 0.0f) {
        std::cerr << "Sigma values must be positive." << std::endl;
        return EXIT_FAILURE;
    }

    if (measuredGpuRuns <= 0) {
        std::cerr << "Number of measured GPU runs must be positive." << std::endl;
        return EXIT_FAILURE;
    }

    if (warmupGpuRuns < 0) {
        std::cerr << "Number of warm-up GPU runs cannot be negative." << std::endl;
        return EXIT_FAILURE;
    }

    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));

    if (deviceCount <= 0) {
        std::cerr << "No CUDA device found." << std::endl;
        return EXIT_FAILURE;
    }

    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));

    std::vector<unsigned char> input;
    int width = 0;
    int height = 0;

    if (!readBMPGray(inputFile, input, width, height)) {
        return EXIT_FAILURE;
    }

    std::vector<unsigned char> cpuOutput;
    std::vector<unsigned char> gpuOutput;

    const auto cpuStart = std::chrono::high_resolution_clock::now();

    bilateralCPU(
        input,
        cpuOutput,
        width,
        height,
        sigmaD,
        sigmaR
    );

    const auto cpuStop = std::chrono::high_resolution_clock::now();

    const double cpuMs =
        std::chrono::duration<double, std::milli>(cpuStop - cpuStart).count();

    GpuTimingStats gpuStats{};

    bilateralGPU(
        input,
        gpuOutput,
        width,
        height,
        sigmaD,
        sigmaR,
        measuredGpuRuns,
        warmupGpuRuns,
        gpuStats
    );

    if (!writeBMPGray24("output_cpu.bmp", cpuOutput, width, height)) {
        return EXIT_FAILURE;
    }

    if (!writeBMPGray24("output_gpu.bmp", gpuOutput, width, height)) {
        return EXIT_FAILURE;
    }

    int maxAbsDiff = 0;
    double meanAbsDiff = 0.0;

    compareImages(cpuOutput, gpuOutput, maxAbsDiff, meanAbsDiff);

    std::cout << std::fixed << std::setprecision(6);

    std::cout << "CUDA device: " << prop.name << std::endl;
    std::cout << "Input image: " << inputFile << std::endl;
    std::cout << "Image size: " << width << " x " << height << std::endl;
    std::cout << "sigma_d: " << sigmaD << std::endl;
    std::cout << "sigma_r: " << sigmaR << std::endl;
    std::cout << "GPU warm-up runs: " << warmupGpuRuns << std::endl;
    std::cout << "GPU measured runs: " << measuredGpuRuns << std::endl;
    std::cout << std::endl;

    std::cout << "CPU processing time: " << cpuMs << " ms" << std::endl;
    std::cout << "GPU setup and host-to-device time: "
              << gpuStats.setupAndCopyToDeviceMs << " ms" << std::endl;
    std::cout << "GPU kernel average time: "
              << gpuStats.kernelAvgMs << " ms" << std::endl;
    std::cout << "GPU kernel min time: "
              << gpuStats.kernelMinMs << " ms" << std::endl;
    std::cout << "GPU kernel max time: "
              << gpuStats.kernelMaxMs << " ms" << std::endl;
    std::cout << "GPU kernel std time: "
              << gpuStats.kernelStdMs << " ms" << std::endl;
    std::cout << "GPU device-to-host time: "
              << gpuStats.copyToHostMs << " ms" << std::endl;
    std::cout << "GPU estimated total time for one processing: "
              << gpuStats.estimatedTotalOneRunMs << " ms" << std::endl;

    std::cout << std::endl;
    std::cout << "CPU/GPU max absolute difference: " << maxAbsDiff << std::endl;
    std::cout << "CPU/GPU mean absolute difference: " << meanAbsDiff << std::endl;

    std::cout << std::endl;
    std::cout << "Saved files:" << std::endl;
    std::cout << "  output_cpu.bmp" << std::endl;
    std::cout << "  output_gpu.bmp" << std::endl;

    return EXIT_SUCCESS;
}
