from setuptools import setup, find_packages

setup(
    name='ros2_numpy',
    version='0.1.0',
    packages=find_packages(),  # Automatically find your package(s)
    install_requires=[
        'numpy',  # Add your dependencies here
        'opencv-python'
    ],
    author='William Engel',
    author_email='william.engel@mdynamix.de',
    description='Converts between common ROS message types and NumPy arrays for easier data manipulation.',  
    long_description=open('README.md').read(),  # Optional, but good practice
    long_description_content_type="text/markdown",  # If you're using Markdown in your README
    url='https://github.com/william-mx/ros2_numpy', 
    license='MIT', 
    classifiers=[  # Optional, but helps with discoverability
        'Development Status :: 3 - Alpha',  # Or "4 - Beta", "5 - Production/Stable"
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',  # Or your chosen license
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Topic :: Software Development :: Libraries',
    ],
)
