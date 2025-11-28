from mininet.topo import Topo
from mininet.net import Mininet
from mininet.node import RemoteController
from mininet.cli import CLI
from mininet.log import setLogLevel

class ProjectTopology(Topo):
    "SDN Project Topology: 1 Switch, 4 Hosts"

    def build(self):
        # Add a single switch
        s1 = self.addSwitch('s1')

        # Add 6 hosts
        h1 = self.addHost('h1', ip='10.0.0.1') # Sentiment Server
        h2 = self.addHost('h2', ip='10.0.0.2') # Tumor Server
        h3 = self.addHost('h3', ip='10.0.0.3') # Sentiment Client
        h4 = self.addHost('h4', ip='10.0.0.4') # Tumor Client
        h5 = self.addHost('h5', ip='10.0.0.5') # Object Detection Server
        h6 = self.addHost('h6', ip='10.0.0.6') # Object Detection Client

        # Connect hosts to the switch
        self.addLink(h1, s1)
        self.addLink(h2, s1)
        self.addLink(h3, s1)
        self.addLink(h4, s1)
        self.addLink(h5, s1)
        self.addLink(h6, s1)


# topos = {'mytopo': MyTopo}


def run():
    topo = ProjectTopology()
    # Connect to a remote controller (Ryu running on localhost:6633)
    net = Mininet(topo=topo, controller=RemoteController)
    
    print("[*] Starting Network")
    net.start()
    
    print("[*] Testing Connectivity (Ping All)")
    net.pingAll()
    
    print("[*] Starting CLI")
    CLI(net)
    
    print("[*] Stopping Network")
    net.stop()

if __name__ == '__main__':
    setLogLevel('info')
    run()
