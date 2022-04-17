python setup.py install --record files.txt
xargs rm -rf < files.txt
rm -rf ~/anaconda3/envs/stonne/lib/python3.7/site-packages/torch_stonne-0.0.0-py3.7-linux-x86_64.egg/
rm files.txt
