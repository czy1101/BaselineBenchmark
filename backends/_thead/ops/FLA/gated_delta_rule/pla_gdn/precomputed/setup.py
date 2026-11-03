from pathlib import Path
from setuptools import setup,Extension
from ppu_build import PPUBuildExtension
p=Path(__file__).resolve().parent
setup(name='pla_precomputed',ext_modules=[Extension('pla_precomputed',
 sources=['host.cpp']+[str(x) for x in sorted(p.glob('pre_*.cu'))],
 include_dirs=[str(p),str(p/'src'),str(p/'src/decode/k-last')],
 extra_compile_args={'cxx':['-O3'],'hgcc':[
                    '-O3',
                    '-std=c++17',
                    '--expt-relaxed-constexpr',
                    '--expt-extended-lambda',
                    '--use_fast_math',
                    '-lineinfo',
                    # '-mllvm', '-ppu-simt-branch=false',
                    '-mllvm', '-ppu-patch-fence-ppu=false',
                    '-mllvm', '-wno-loop-miss-transform',
                    '-mllvm', '-ppu-cg-to-kp1=true',
                    '-mllvm', '-ppu-fix-uninit=true',
                    '-mllvm', '-ppu-force-warpage=true',
                    '-mllvm', '-sort-copy-before-coalesce=true',
                    '-mllvm', '-ppu-eliminate-early-v2s=true',
                    '-mllvm', '-ppu-pref-fma-reuse=true',
                    '--threads', '2',
                    # Real device targets, not the vm_* virtual IR ones: the
                    # PPU runtime does not JIT, so a vm_* (or missing) -arch
                    # makes every launch fail with 801 operation-not-supported
                    # while hggcDeviceSynchronize() still returns success.
                    # Emit one native image flag per supported PPU generation.
                    '-arch=ppu_10', '-arch=ppu_15'
                ]})],
 cmdclass={'build_ext':PPUBuildExtension})
