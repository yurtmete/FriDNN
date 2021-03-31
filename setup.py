from setuptools import setup

setup(
    name='FriDNN',
    version='0.0.0',
    description='Fricative Detection Using 1D CNNs',
    url='https://github.com/yurtmete/FriDNN',
    author='Metehan Yurt',
    author_email='metehanyurt@gmail.com',
    license='MIT License',
    packages=[
        'train_test'
    ],
    install_requires=[
        'Keras==2.2.4',
        'h5py==2.10.0',
        'numpy==1.15.4',
        'scipy==1.5.3',
        'scikit-learn==0.20.0',
        'SoundFile==0.10.2',
        'tensorflow-gpu==1.14.0',
        'tensorboard==2.4.1',
    ],
    zip_safe=False
)
