import os

from glob import glob
from setuptools import setup, find_packages

package_name = 'hex_ros_policy_client'


def get_files(tar: str, src: str):
    """Collect data files recursively, keeping the source tree layout.

    Args:
        tar (str): Destination directory inside the install space.
        src (str): Source directory relative to the package root.

    Returns:
        list: Pairs of (destination directory, source file).
    """
    all_paths = glob(f'{src}/*')

    data_files = []
    for path in all_paths:
        if os.path.isfile(path):
            data_files.append((tar, [path]))
        elif os.path.isdir(path):
            sub_files = get_files(f'{tar}/{os.path.basename(path)}', path)
            data_files.extend(sub_files)

    return data_files


setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
         ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name, ['README.md', 'README_cn.md']),
        *get_files('share/' + package_name, 'launch/ros2'),
        *get_files('share/' + package_name + '/config/ros2', 'config/ros2'),
    ],
    install_requires=['setuptools'],
    tests_require=['pytest'],
    zip_safe=True,
    maintainer='Dong Zhaorui',
    maintainer_email='dzr159@gmail.com',
    description=(
        'Policy client for imitation-learning deployment: '
        'publishes observations and executes action chunks.'),
    license='Apache-2.0',
    entry_points={
        'console_scripts': [
            'dual_manip_policy_client = '
            'hex_ros_policy_client.dual_manip_policy_client:main',
        ],
    },
)
