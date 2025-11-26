import triton
import triton.language as tl


@triton.jit
def say_hello_kernel():
    pid = tl.program_id(0)
    print("Hello, World from Triton program id:", pid)


def main():
    num_programs = 2
    say_hello_kernel[(num_programs,)](num_warps=1)


if __name__ == "__main__":
    main()
