# SDN IoT Project

This project implements an SDN-based IoT network simulation using Mininet and Ryu controller, featuring various AI/ML applications like object detection, sentiment analysis, and tumor detection.

## Prerequisites & Installation

### 1. Install Mininet
Clone the Mininet repository and run the installation script:
```bash
git clone https://github.com/mininet/mininet
mininet/util/install.sh -a
```

### 2. Install Ryu Controller
Install Ryu using pip:
```bash
sudo pip3 install ryu
```

## Project Setup

### 1. Clone the Repository
Clone this project using either HTTPS or SSH:

**HTTPS:**
```bash
git clone https://github.com/ahmedgaboelnaga/sdn-iot.git
```

**SSH:**
```bash
git clone git@github.com:ahmedgaboelnaga/sdn-iot.git
```

### 2. Install Dependencies

First, install the CPU version of PyTorch:
```bash
sudo pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

Then, install the remaining Python requirements:
```bash
cd sdn-iot
sudo pip3 install -r requirements.txt
```

## Usage

### 1. Start the Ryu Controller
Run the simple switch controller:
```bash
ryu-manager simple_switch.py
```

### 2. Start Mininet Topology
In a separate terminal, start the Mininet virtual network:
```bash
sudo python3 topology.py
```

### 3. Open Host Terminals
Inside the Mininet CLI, open terminals for the hosts (h1 to h6):
```bash
mininet> xterm h1 h2 h3 h4 h5 h6
```

### 4. Run Servers and Clients
You can run the following applications on the host terminals you just opened. Typically, run the server on one host and the client on another.

#### Object Detection
*   **Server:** `python3 object_detection_server.py`
*   **Client:** `python3 object_detection_client.py`

#### Sentiment Analysis
*   **Server:** `python3 sentiment_server.py`
*   **Client:** `python3 sentiment_client.py`

#### Tumor Detection
*   **Server:** `python3 tumor_server.py`
*   **Client:** `python3 tumor_client.py`
