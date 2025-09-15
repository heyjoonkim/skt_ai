export CUDA_VISIBLE_DEVICES=0
tensor_parallel_size=1

# VLLM_LOGGING_LEVEL=DEBUG

python -m train.build_dataset \
    tensor_parallel_size=$tensor_parallel_size


curl -d "Done [Train (DPO) : build_dataset] process... Please Check..." ntfy.sh/hjkim-experiments
