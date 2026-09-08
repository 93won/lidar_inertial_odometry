# Tightly-Coupled LiDAR-Inertial_Odometry

[MIT Licence](LICENSE)

## Documentation

- [Quick Start](docs/Quick_Start.md)
- [Release Notes — v0.1.0](docs/Release_Notes_v0.1.0.md)
- [ROS 2 Wrapper](https://github.com/93won/lio_ros_wrapper) — requires updates for the v0.1.0 core.

## References

### Surfel-LIO

Seungwon Choi, Dong-Gyu Park, Seo-Yeon Hwang, and Tae-Wan Kim. [Surfel-LIO: Fast LiDAR-Inertial Odometry with Pre-computed Surfels and Hierarchical Z-order Voxel Hashing](https://arxiv.org/abs/2512.03397), 2025.

The paper describes the original surfel-based implementation; the current version uses an incremental voxel KD-tree.

```bibtex
@misc{choi2025surfellio,
  title={Surfel-LIO: Fast LiDAR-Inertial Odometry with Pre-computed Surfels and Hierarchical Z-order Voxel Hashing},
  author={Seungwon Choi and Dong-Gyu Park and Seo-Yeon Hwang and Tae-Wan Kim},
  year={2025},
  eprint={2512.03397},
  archivePrefix={arXiv},
  primaryClass={cs.RO},
  url={https://arxiv.org/abs/2512.03397}
}
```

### Probabilistic Kernel Optimization

The repository includes this method as a separate module; the current iterated ESKF update does not use it.

```bibtex
@article{choi2025pko,
  title={Probabilistic Kernel Optimization for Robust State Estimation},
  author={Choi, Seungwon and Kim, Tae-Wan},
  journal={IEEE Robotics and Automation Letters},
  volume={10},
  number={3},
  pages={2998--3005},
  year={2025},
  publisher={IEEE}
}
```
