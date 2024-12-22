from setuptools import setup, find_packages

setup(
    name="iot-sensor-data-monitoring",
    version="1.0.0",
    description="Predict failures in IoT sensor data monitoring",
    author="Atouf Ayoub",
    author_email="atouf.ayoub.1@gmail.com",
    url="https://github.com/atoufayoub/IOT_sensor_data_monitoring",
    license="MIT",
    packages=find_packages(),
    install_requires=[
        "fastapi==0.115.4",
        "uvicorn==0.32.0",
        "joblib==1.2.0",
        "matplotlib==3.5.3",
        "numpy==1.23.5",
        "pandas==2.2.3",
        "psutil==6.0.0",
        "pydantic==1.10.9",
        "scikit_learn==1.1.3",
        "scikit_optimize==0.10.2",
        "scipy==1.14.1",
        "seaborn==0.13.2",
        "mlflow==2.16.2",
        "watchfiles==0.24.0",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
        "License :: OSI Approved :: MIT License",
    ],
    python_requires="==3.10.12",
)
