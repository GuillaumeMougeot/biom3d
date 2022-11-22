from setuptools import setup, find_namespace_packages

install_deps = [
    "torch>1.10.0",
    "tqdm",
    "scikit-image>=0.14",
    "scipy",
    "numpy",
    "sklearn",
    "SimpleITK",
    "pandas",
    "tifffile", 
    "matplotlib",
    "tensorboard",
    "PyYAML",
    "torchio",
    "protobuf"
]

docs_deps = [
    'sphinx>=3.0',
    'sphinxcontrib-apidoc',
    'sphinx_rtd_theme',
]

gui_deps = [
    "paramiko",
    "omero-py",
    "netcat"
]


setup(name='biom3d',
    packages=find_namespace_packages(include=["biom3d", "biom3d.*"]),
    version='2022.11.22',
    description='Biom3d. Framework for easy-to-use biomedical image segmentation.',
    url='https://github.com/GuillaumeMougeot/biom3d',
    author='Guillaume Mougeot',
    author_email='guillaume.mougeot@laposte.net',
    license='Apache License Version 2.0, January 2004',
    install_requires=install_deps,
    entry_points={
    # 'console_scripts': [
    #     'biom3d = biom3d.__main__:main']
    },
    extras_require = {
        'docs': docs_deps,
        'gui': gui_deps,
        'all': gui_deps + docs_deps,
    },
    keywords=['deep learning', 'image segmentation', 'medical image analysis',
            'medical image segmentation', 'biological image segmentation', 'bio-imaging']
)