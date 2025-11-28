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

        # Add 4 hosts
        h1 = self.addHost('h1', ip='10.0.0.1')
        h2 = self.addHost('h2', ip='10.0.0.2')
        h3 = self.addHost('h3', ip='10.0.0.3')
        h4 = self.addHost('h4', ip='10.0.0.4')

        # Connect hosts to the switch
        self.addLink(h1, s1)
        self.addLink(h2, s1)
        self.addLink(h3, s1)
        self.addLink(h4, s1)


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
