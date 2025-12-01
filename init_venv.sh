#!/bin/bash

# Script to initialize Python virtual environment and install requirements
# Author: Auto-generated
# Date: 2025-11-29

set -e  # Exit on error

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Configuration
VENV_DIR="venv"
REQUIREMENTS_FILE="requirements.txt"
PYTHON_CMD="python3"

echo -e "${GREEN}=== Python Virtual Environment Initialization ===${NC}\n"

# Check if Python is installed
if ! command -v $PYTHON_CMD &> /dev/null; then
    echo -e "${RED}Error: Python3 is not installed or not in PATH${NC}"
    exit 1
fi

echo -e "${GREEN}✓${NC} Python found: $($PYTHON_CMD --version)"

# Check if requirements.txt exists
if [ ! -f "$REQUIREMENTS_FILE" ]; then
    echo -e "${RED}Error: $REQUIREMENTS_FILE not found${NC}"
    exit 1
fi

echo -e "${GREEN}✓${NC} Requirements file found: $REQUIREMENTS_FILE"

# Check if virtual environment already exists
if [ -d "$VENV_DIR" ]; then
    echo -e "${YELLOW}⚠${NC}  Virtual environment already exists at '$VENV_DIR'"
    read -p "Do you want to remove it and create a new one? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo -e "${YELLOW}Removing existing virtual environment...${NC}"
        rm -rf "$VENV_DIR"
    else
        echo -e "${YELLOW}Using existing virtual environment${NC}"
    fi
fi

# Create virtual environment if it doesn't exist
if [ ! -d "$VENV_DIR" ]; then
    echo -e "\n${GREEN}Creating virtual environment...${NC}"
    $PYTHON_CMD -m venv "$VENV_DIR"
    echo -e "${GREEN}✓${NC} Virtual environment created at '$VENV_DIR'"
fi

# Activate virtual environment
echo -e "\n${GREEN}Activating virtual environment...${NC}"
source "$VENV_DIR/bin/activate"

# Verify activation
if [ -z "$VIRTUAL_ENV" ]; then
    echo -e "${RED}Error: Failed to activate virtual environment${NC}"
    exit 1
fi

echo -e "${GREEN}✓${NC} Virtual environment activated"

# Upgrade pip
echo -e "\n${GREEN}Upgrading pip...${NC}"
pip install --upgrade pip

# Install requirements
echo -e "\n${GREEN}Installing requirements from $REQUIREMENTS_FILE...${NC}"
pip install -r "$REQUIREMENTS_FILE"

# Summary
echo -e "\n${GREEN}=== Installation Complete ===${NC}"
echo -e "${GREEN}✓${NC} Virtual environment: $VENV_DIR"
echo -e "${GREEN}✓${NC} Python version: $($PYTHON_CMD --version)"
echo -e "${GREEN}✓${NC} Pip version: $(pip --version)"
echo -e "${GREEN}✓${NC} Installed packages: $(pip list | wc -l) packages"

echo -e "\n${YELLOW}To activate the virtual environment in the future, run:${NC}"
echo -e "  ${GREEN}source $VENV_DIR/bin/activate${NC}"

echo -e "\n${YELLOW}To deactivate the virtual environment, run:${NC}"
echo -e "  ${GREEN}deactivate${NC}"
