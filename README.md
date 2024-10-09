# Reinforcement Learning Course

This repository contains code and projects related to the Reinforcement Learning course. Follow the steps below to set up your environment and get started with the projects.

## Table of Contents

- [Installation](#installation)
- [License](#license)

## Installation

Follow these steps to set up your environment:

1. **Install PyCharm or any other IDE you are familiar with.**
2. **Install Python (version 3.11 or newer).**
3. **Clone the repository using your chosen IDE, Git client, or by downloading the repository as a zip file:**
   ```bash
   git clone https://github.com/VolDonets/reinforcement_learining.git
4. **Open the cloned project and select a virtual environment (venv) for the libraries to avoid cluttering system-wide libraries.**
    ```bash on Windows
    python -m venv venv
    .\venv\Scripts\activate
5. **Install dependecies. You need swig to compile some of dependencies. You install with choco https://chocolatey.org/**
    choco install swig
    pip install -r requirements.txt
    pip install "gymnasium[atari,accept-rom-license]"
5. **IMPORTANT: The folder of interest is gymnasium_version, which contains the updated code relevant to this course.**

## License

This project is licensed under the MIT License.
