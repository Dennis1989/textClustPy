pip3 uninstall textClustPy -y

python3 setup.py sdist bdist_wheel
cd dist
pip3 install textClustPy-0.0.1-py3-none-any.whl 
