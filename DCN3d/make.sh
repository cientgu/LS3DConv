rm -rf build/
rm -rf DCN.egg-info
rm -rf dist
python setup.py build install --prefix=~/python_lib
