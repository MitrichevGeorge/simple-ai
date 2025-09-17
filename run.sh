#!/bin/bash

VENV_DIR="venv"
REQ_FILE="requirements.txt"
MAIN_SCRIPT="smp.py"

if [ ! -d "$VENV_DIR" ]; then
    python3 -m venv $VENV_DIR

    source $VENV_DIR/bin/activate
    pip install --upgrade pip
    pip install -r $REQ_FILE

    cat > "$0" <<EOL
#!/bin/bash
source $VENV_DIR/bin/activate
python $MAIN_SCRIPT
EOL
    chmod +x "$0"

    python $MAIN_SCRIPT
else
    source $VENV_DIR/bin/activate
    python $MAIN_SCRIPT
fi

