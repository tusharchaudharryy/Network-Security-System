<<<<<<< HEAD
from setuptools import setup, find_packages
from typing import List
HYPHEN_E_DOT = '-e .'

def get_requirements(file_path:str)->List[str]:
    '''
    This function will return the list of requirements
    '''
    requirements = []
    with open(file_path) as file_obj:
        requirements = file_obj.readlines()
        requirements = [req.replace('\n', '') for req in requirements]

        if HYPHEN_E_DOT in requirements:
            requirements.remove(HYPHEN_E_DOT)

    return requirements

setup(
    name='ML_Project',
    version='0.0.1',
    author='Tushar Chaudhary',
    author_email='chaudharytushar477@gmail.com',
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt')

=======
from setuptools import setup, find_packages
from typing import List
HYPHEN_E_DOT = '-e .'

def get_requirements(file_path:str)->List[str]:
    '''
    This function will return the list of requirements
    '''
    requirements = []
    with open(file_path) as file_obj:
        requirements = file_obj.readlines()
        requirements = [req.replace('\n', '') for req in requirements]

        if HYPHEN_E_DOT in requirements:
            requirements.remove(HYPHEN_E_DOT)

    return requirements

setup(
    name='ML_Project',
    version='0.0.1',
    author='Tushar Chaudhary',
    author_email='chaudharytushar477@gmail.com',
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt')

>>>>>>> 568bd63 (New Commits)
)