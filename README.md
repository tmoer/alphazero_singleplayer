# Single player Alpha Zero implementation

Forked from https://github.com/tmoer/alphazero_singleplayer.
Mainly just adding docs so I can remember how to use it.

# Usage
Tested with Ubuntu 21.10 python 3.7 (I used pyenv to run old python). Note that
an old python version is required to be able to use the old Tensorflow version
(1.x) required by this code.

- [installing pyenv](http://codingadventures.org/2020/08/30/how-to-install-pyenv-in-ubuntu/)

```sh
pyenv install 3.7.12
pyenv local 3.7.12
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python alphazero.py --n_ep 50 --n_mcts 10
```