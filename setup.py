'''
it is essential part of the packaging and distributed python project. It is used by setup tools to define the configuration of the project such as its metadata, dependencies, and more.
'''

from setuptools import setup, find_packages
from typing import List

def get_requirements()-> List[str]:
    '''
    This function returns a list of requirements from the requirements.txt file.
    '''
    try:
        requirements_list:List[str] = []
        # Read the requirements.txt file
        with open('requirements.txt', 'r') as file:
            # Read each line in the file
            lines = file.readlines()
            for line in lines:
                # Strip whitespace and comments
                requirement = line.strip()
                if requirement and requirement!='-e .':
                    requirements_list.append(requirement)
        return requirements_list
    except FileNotFoundError:
        print("requirements.txt file not found. Please ensure it exists in the project directory.")
    except Exception as e:
        print(f"An error occurred while reading requirements.txt: {e}")

setup(
    name='Network-Security',
    version='0.0.1',
    author='Laxman Gouda',
    author_email='laxman.sg0104@gmail.com',
    packages=find_packages(),
    install_requires=get_requirements()
)