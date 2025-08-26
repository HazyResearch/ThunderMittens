#define RUN_ATTN_OBJ_C
#ifdef RUN_ATTN_OBJ_C

#pragma once

#include <MetalPerformanceShaders/MetalPerformanceShaders.h>
#import <Foundation/Foundation.h>
#import <QuartzCore/QuartzCore.h>
//#include "../../testing/test_utils.h"

#define INIT_TIMER \
    CFTimeInterval startTime, endTime, totalTime;

#define START_TIMER \
    startTime = CACurrentMediaTime();

#define END_TIMER \
    endTime = CACurrentMediaTime(); \
    totalTime = endTime - startTime;

#define GET_TIME \
    totalTime

#define INIT_CAPTURE \
    MTLCaptureManager *captureManager;

#define CAPTURE_START \
    NSError *CAPTURE_error; \
    captureManager = [MTLCaptureManager sharedCaptureManager]; \
    MTLCaptureDescriptor *captureDescriptor = [MTLCaptureDescriptor alloc]; \
    captureDescriptor.captureObject = device; \
    captureDescriptor.destination = MTLCaptureDestinationDeveloperTools; \
    [captureManager startCaptureWithDescriptor:captureDescriptor error:&CAPTURE_error]; \
    [NSThread sleepForTimeInterval:1.0f]; \

#define CAPTURE_END \
    [captureManager stopCapture];
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>



bfloat16_t floatToBfloat16(float input) {
    uint32_t floatBits = *((uint32_t*)&input);
    uint16_t floatBits16 = (uint16_t)(floatBits >> 16);
    bfloat16_t bf16Bits = *((bfloat16_t*)(&floatBits16));

    return bf16Bits;
}

bfloat16_t *loadTensorSegment(bfloat16_t *sourceData, NSUInteger offset, NSUInteger tensorSize) {
    bfloat16_t *tensor = (bfloat16_t *)malloc(tensorSize * sizeof(bfloat16_t));
    if (tensor == NULL) {
        NSLog(@"Failed to allocate memory for tensor.");
        return NULL;
    }
    memcpy(tensor, sourceData + offset, tensorSize * sizeof(bfloat16_t));
    return tensor;
}

bfloat16_t *loadAllTensorsFromFile(NSString *filePath, NSUInteger totalSize) {
    NSError *error = nil;
    NSString *fileContents = [NSString stringWithContentsOfFile:filePath
                                                       encoding:NSUTF8StringEncoding
                                                          error:&error];
    if (error) {
        NSLog(@"Error reading file: %@", [error localizedDescription]);
        return NULL;
    }

    NSArray<NSString *> *components = [fileContents componentsSeparatedByString:@" "];
    if (components.count < totalSize) {
        NSLog(@"File does not contain enough data.");
        return NULL;
    }
    
    bfloat16_t *allData = (bfloat16_t *)malloc(totalSize * sizeof(bfloat16_t));
    for (NSUInteger i = 0; i < totalSize; i++) {
        allData[i] = floatToBfloat16([components[i] floatValue]);
    }

    return allData;
}

void createAndBindMTLBuffer(id<MTLDevice> device, bfloat16_t *data, NSUInteger dataSize, id<MTLBuffer> *bufferOut) {
    *bufferOut = [device newBufferWithBytes:data
                                     length:dataSize * sizeof(bfloat16_t)
                                    options:MTLResourceStorageModeShared];
    if (*bufferOut == nil) {
        NSLog(@"Failed to create MTLBuffer.");
    }
}

double launch_kernel(id<MTLDevice> device, id<MTLCommandQueue> command_queue,
                            id<MTLBuffer> q, id<MTLBuffer> k, id<MTLBuffer> v, id<MTLBuffer> o_res,
                            NSString* kernel_name,
                            int B, int H, int N, int D,
                     int NUM_WORKERS, int TILE_ROW) {
    INIT_TIMER
    NSError *error;
    id<MTLCommandBuffer> command_buffer = [command_queue commandBuffer];
    assert(command_buffer != nil);
    id<MTLComputeCommandEncoder> compute_encoder = [command_buffer computeCommandEncoder];
    assert(compute_encoder != nil);
    id<MTLLibrary> default_library = [device newDefaultLibrary];
    assert(default_library != nil);
    id<MTLFunction> kernel = [default_library newFunctionWithName:kernel_name];
    assert(kernel != nil);
    id<MTLComputePipelineState> kernelPSO = [device newComputePipelineStateWithFunction:kernel error:&error];
    assert(kernelPSO != nil);

    MTLSize threadgroup_size = MTLSizeMake(32*NUM_WORKERS, 1, 1);
    MTLSize grid = MTLSizeMake(N / (TILE_ROW*NUM_WORKERS), H, B);

    [compute_encoder setComputePipelineState:kernelPSO];
    [compute_encoder setBuffer:q offset:0 atIndex:0];
    [compute_encoder setBuffer:k offset:0 atIndex:1];
    [compute_encoder setBuffer:v offset:0 atIndex:2];
    [compute_encoder setBuffer:o_res offset:0 atIndex:3];
    [compute_encoder setBytes:&N length:sizeof(unsigned) atIndex:4];
    [compute_encoder setBytes:&H length:sizeof(unsigned) atIndex:5];
    
    [compute_encoder dispatchThreadgroups:grid threadsPerThreadgroup:threadgroup_size];
    [compute_encoder endEncoding];
    START_TIMER
    [command_buffer commit];
    [command_buffer waitUntilCompleted];
    END_TIMER
    
    return GET_TIME;
}

void check_attn_correctness(id<MTLBuffer> _o_ref, id<MTLBuffer> _o_res, int B, int H, int N, int D) {
    const float epsilon = 5e-2;
    NSError *error = nil;
    bfloat16_t* o_ref = [_o_ref contents];
    bfloat16_t* o_res = [_o_res contents];
    NSString *currentPath = [[NSFileManager defaultManager] currentDirectoryPath];
    NSString* basePath = [NSString stringWithFormat:@"%@/ThunderMittens/kernels/attn_fwd/correctness/", currentPath];
    for (int b = 0; b < B; b++) {
    for (int h = 0; h < H; h++) {
    for (int n = 0; n < N; n++) {
    for (int d = 0; d < D; d++) {
        float ref = o_ref[b * (H*N*D) + h * (N*D) + n * (D) + d];

        float res = (float)o_res[b * (H*N*D) + h * (N*D) + n * (D) + d];
        if (fabs(ref - res) > epsilon) {
            NSLog(@"mismatch  : (%d, %d, %d, %d):\n"
                  @" dims     : (%d, %d, %d, %d)\n"
                  @" reference: %.3f\n"
                  @" result   : %.3f",
                  b, h, n, d,
                  B, H, N, D,
                  ref, res);
            
            NSMutableString* log1 = [NSMutableString stringWithString:@""];
            float avg_diff = 0.f;
            for (int n = 0; n < N; n++) {
                for (int d = 0; d < D; d++) {
                    float ref = (float)o_ref[b * (H*N*D) + h * (N*D) + n * (D) + d];
                    float res = (float)o_res[b * (H*N*D) + h * (N*D) + n * (D) + d];
                    float diff = fabs(ref - res);
                    avg_diff += diff;
                    [log1 appendFormat: @"%.3f ", diff];
                }
                [log1 appendString: @"\n"];
            }
            NSString* diffPath = [NSString stringWithFormat:@"%@diff.txt", basePath];
            [log1 writeToFile:diffPath atomically:YES encoding:NSUTF8StringEncoding error:&error];
            avg_diff /= (N * D);
            
            NSMutableString* log2 = [NSMutableString stringWithString:@""];
            float avg_ref = 0.f;
            for (int n = 0; n < N; n++) {
                for (int d = 0; d < D; d++) {
                    float ref = (float)o_ref[b * (H*N*D) + h * (N*D) + n * (D) + d];
                    [log2 appendFormat: @"%.3f ", ref];
                    avg_ref += fabs(ref);
                }
                [log2 appendString: @"\n"];
            }
            NSString* refPath = [NSString stringWithFormat:@"%@ref.txt", basePath];
            [log2 writeToFile:refPath atomically:YES encoding:NSUTF8StringEncoding error:&error];
            avg_ref /= (N * D);
            float avg_res = 0.f;
            NSMutableString* log3 = [NSMutableString stringWithString:@""];
            for (int n = 0; n < N; n++) {
                for (int d = 0; d < D; d++) {
                    float res = (float)o_res[b * (H*N*D) + h * (N*D) + n * (D) + d];
                    avg_res += fabs(res);
                    [log3 appendFormat: @"%.3f ", res];
                }
                [log3 appendString: @"\n"];
            }
            NSString* resPath = [NSString stringWithFormat:@"%@res.txt", basePath];
            [log3 writeToFile:resPath atomically:YES encoding:NSUTF8StringEncoding error:&error];
            avg_res /= (N * D);
            
            NSLog(@"avg diff magnitude  : %.5f\n"
                  @"avg ref  magnitude  : %.5f\n"
                  @"avg res  magnitude  : %.5f\n",
                  avg_diff, avg_ref, avg_res);
            return;
            assert(false);
        }
    }
    }
    }
    }
    NSLog(@"Correctness passed!");
}





int main(int argc, const char * argv[]) {
    NSString *currentPath = [[NSFileManager defaultManager] currentDirectoryPath];
    NSString *filePath = [NSString stringWithFormat:@"%@/ThunderMittens/kernels/attn_fwd/correctness/randn_4_4_1024_128.txt", currentPath];
    NSLog(@"%@", filePath);
    @autoreleasepool {
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        id<MTLCommandQueue> command_queue = [device newCommandQueue];
        if (!device) {
            NSLog(@"Metal is not supported on this device.");
            return -1;
        }

        // Parameterize B, H, N, D (example values)
        NSUInteger B = 4; // Batch size
        NSUInteger H = 4; // Number of heads
        NSUInteger N = 1024; // Sequence length
        NSUInteger D = 128; // Feature dimension

        int TILE_ROW = 8;
        int NUM_WORKERS = 1;

        NSUInteger tensorSizePerTensor = B * H * N * D;
        NSUInteger totalSize = 4 * tensorSizePerTensor; // Total size for Q, K, V, O

        
        bfloat16_t *allData = loadAllTensorsFromFile(filePath, totalSize);
        if (allData == NULL) {
            NSLog(@"Failed to load tensor data.");
            return -1;
        }

        // Split the data into separate tensors
        bfloat16_t *qData = loadTensorSegment(allData, 0 * tensorSizePerTensor, tensorSizePerTensor);
        bfloat16_t *kData = loadTensorSegment(allData, 1 * tensorSizePerTensor, tensorSizePerTensor);
        bfloat16_t *vData = loadTensorSegment(allData, 2 * tensorSizePerTensor, tensorSizePerTensor);
        bfloat16_t *oData = loadTensorSegment(allData, 3 * tensorSizePerTensor, tensorSizePerTensor);

        free(allData); // Free the contiguous data buffer after splitting

        if (qData == NULL || kData == NULL || vData == NULL || oData == NULL) {
            NSLog(@"Failed to split tensor data.");
            return -1;
        }

        id<MTLBuffer> qBuffer, kBuffer, vBuffer, oBuffer;
        createAndBindMTLBuffer(device, qData, tensorSizePerTensor, &qBuffer);
        createAndBindMTLBuffer(device, kData, tensorSizePerTensor, &kBuffer);
        createAndBindMTLBuffer(device, vData, tensorSizePerTensor, &vBuffer);
        createAndBindMTLBuffer(device, oData, tensorSizePerTensor, &oBuffer);

        // Free allocated memory for each tensor
        free(qData);
        free(kData);
        free(vData);
        free(oData);
        id<MTLBuffer> o_res = [device newBufferWithLength:B * H * N * D * sizeof(bfloat16_t) options:MTLResourceStorageModeShared];
 
        NSLog(@"Q Buffer size: %lu bytes", qBuffer.length);
        NSLog(@"K Buffer size: %lu bytes", kBuffer.length);
        NSLog(@"V Buffer size: %lu bytes", vBuffer.length);
        NSLog(@"O Buffer size: %lu bytes", oBuffer.length);
        NSLog(@"Output Buffer size: %lu bytes", o_res.length);
        
        NSString* kernel_name = [NSString stringWithFormat:@"attn_fwd_%d", (int)D];
                
        launch_kernel(device, command_queue, qBuffer, kBuffer, vBuffer, o_res, kernel_name, B, H, N, D, NUM_WORKERS, TILE_ROW);
        check_attn_correctness(oBuffer, o_res, B, H, N, D);
    }
    return 0;
}
#endif
