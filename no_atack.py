from mininet.node import Controller
from mininet.log import setLogLevel, info
from mn_wifi.net import Mininet_wifi
from mn_wifi.link import wmediumd, adhoc
from mn_wifi.wmediumdConnector import interference
import time
import random
import math
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
import numpy as np

def topology():
    net = Mininet_wifi(link=wmediumd, wmediumd_mode=interference)

    info("*** Criando os drones\n")
    drones = []
    num_drones = 5
    for i in range(1, num_drones + 1):
        drone = net.addStation(f'drone{i}', position='0,0,0', wlan=1)
        drones.append(drone)

    info("*** Configurando os nós WiFi\n")
    net.configureWifiNodes()

    info("*** Criando links adhoc entre os drones\n")
    for i, drone in enumerate(drones):
        net.addLink(drone, cls=adhoc, intf=f'{drone.name}-wlan0',
                    ssid='adhocNet', mode='g', channel=5, proto='babel')

    info("*** Definindo o modelo de propagação\n")
    net.setPropagationModel(model="logDistance", exp=4)

    info("*** Construindo a rede\n")
    net.build()
    net.start()

    area_size = 30
    formation_center = [area_size / 2, area_size / 2]
    radius = 5
    angle_step = 360 / num_drones

    for i, drone in enumerate(drones):
        angle = math.radians(i * angle_step)
        x = formation_center[0] + radius * math.cos(angle)
        y = formation_center[1] + radius * math.sin(angle)
        drone.params['position'] = [x, y, 0]

    target_position = [random.uniform(0, area_size), random.uniform(0, area_size), 0]

    plt.ion()
    fig, ax = plt.subplots()
    ax.set_xlim(0, area_size)
    ax.set_ylim(0, area_size)
    ax.set_title("Monitoramento dos Drones")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    drone_positions, = ax.plot([], [], 'bo')
    target_position_marker, = ax.plot([], [], 'ro')
    ax.legend()

    def calculate_circular_position(index, center, radius, angle_step):
        angle = math.radians(index * angle_step)
        x = center[0] + radius * math.cos(angle)
        y = center[1] + radius * math.sin(angle)
        return [x, y, 0]

    def calculate_distance(pos1, pos2):
        return ((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)**0.5

    def calculate_rssi(drone_a, drone_b):
        TX_POWER = 14
        PATH_LOSS_EXPONENT = 4
        REFERENCE_DISTANCE = 1
        PATH_LOSS_0 = 40
        pos_a = drone_a.params['position']
        pos_b = drone_b.params['position']
        distance = calculate_distance(pos_a, pos_b)
        if distance < REFERENCE_DISTANCE:
            distance = REFERENCE_DISTANCE
        path_loss = PATH_LOSS_0 + 10 * PATH_LOSS_EXPONENT * math.log10(distance / REFERENCE_DISTANCE)
        return TX_POWER - path_loss

    def check_connectivity(drone_a, drone_b):
        signal_strength = calculate_rssi(drone_a, drone_b)
        return signal_strength >= -80, signal_strength

    def update_graph():
        positions = [drone.params['position'] for drone in drones]
        x_coords = [pos[0] for pos in positions]
        y_coords = [pos[1] for pos in positions]
        drone_positions.set_data(x_coords, y_coords)
        target_position_marker.set_data(target_position[0], target_position[1])
        plt.draw()
        plt.pause(0.1)

    def move_drones_in_formation(drones, center, radius, angle_step):
        all_drones_in_position = True
        for i, drone in enumerate(drones):
            target_position = calculate_circular_position(i, center, radius, angle_step)
            current_position = drone.params['position']
            dx = target_position[0] - current_position[0]
            dy = target_position[1] - current_position[1]
            distance = math.sqrt(dx**2 + dy**2)
            if distance > 1:
                step_size = 1
                dx_normalized = dx / distance * step_size
                dy_normalized = dy / distance * step_size
                drone.params['position'][0] += dx_normalized
                drone.params['position'][1] += dy_normalized
                all_drones_in_position = False
        return all_drones_in_position

    for drone in drones:
        drone.params['cumulative_distance'] = 0.0
        drone.params['previous_position'] = drone.params['position'].copy()
    all_positions = []
    total_signal_strength = 0
    signal_strength_count = 0

    mission_active = True
    mission_time = 0
    max_mission_time = 60
    all_connected_time = 0
    target_found = False

    while mission_active and mission_time < max_mission_time:
        info(f"\n\n=== Tempo da missão: {mission_time} segundos ===\n")
        for drone in drones:
            pos = drone.params['position']
            all_positions.append([pos[0], pos[1]])
            current_position = drone.params['position']
            previous_position = drone.params['previous_position']
            distance_moved = calculate_distance(current_position, previous_position)
            drone.params['cumulative_distance'] += distance_moved
            drone.params['previous_position'] = current_position.copy()
            info(f"  {drone.name}: posição [x: {pos[0]:.2f}, y: {pos[1]:.2f}, z: {pos[2]:.2f}]\n")

        info("Conectividade entre drones:\n")
        all_connected = True
        for i in range(len(drones)):
            for j in range(i + 1, len(drones)):
                connected, signal_strength = check_connectivity(drones[i], drones[j])
                if connected:
                    info(f"  {drones[i].name} <-> {drones[j].name}: Conectados (Sinal: {signal_strength:.2f} dBm)\n")
                else:
                    info(f"  {drones[i].name} <-> {drones[j].name}: Desconectados (Sinal: {signal_strength:.2f} dBm)\n")
                    all_connected = False

        if all_connected:
            all_connected_time += 1

        update_graph()
        distance_to_target = min(
            calculate_distance(drone.params['position'], target_position) for drone in drones
        )
        info(f"[PROGRESSO] Distância mínima do alvo: {distance_to_target:.2f} metros\n")

        if distance_to_target < 8 and not target_found:
            info(f"[SUCESSO] O alvo foi encontrado!\n")
            target_found = True

        if target_found:
            all_drones_in_position = move_drones_in_formation(drones, target_position, radius, angle_step)
            if all_drones_in_position:
                info(f"[FINALIZAÇÃO] Drones em formação ao redor do alvo na posição [x: {target_position[0]:.2f}, y: {target_position[1]:.2f}]. Missão encerrada.\n")
                break
        else:
            formation_center[0] += random.uniform(-2, 2)
            formation_center[1] += random.uniform(-2, 2)
            move_drones_in_formation(drones, formation_center, radius, angle_step)

        mission_time += 1
        time.sleep(0.5)

    points = np.array(all_positions)
    if len(points) >= 3:
        hull = ConvexHull(points)
        area_covered = hull.volume
    else:
        area_covered = 0

    info("\n*** MÉTRICAS FINAIS ***\n")
    info(f"Tempo total da missão: {mission_time} segundos\n")
    connectivity_percentage = ((all_connected_time-1) / mission_time) * 100
    info(f"Tempo em que todos os drones estiveram conectados: {all_connected_time-1} segundos ({connectivity_percentage:.2f}%)\n")
    info(f"Área coberta pelos drones: {area_covered:.2f} metros quadrados\n")

    plt.ioff()
    plt.show()
    net.stop()

if __name__ == '__main__':
    setLogLevel('info')
    topology()
