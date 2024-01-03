import streamlit as st
import numpy as np
import cv2
from matplotlib import pyplot as plt
from PIL import Image
import pandas as pd
import networkx as nx
from sklearn.cluster import KMeans
from streamlit_option_menu import option_menu

def extract_colors():
    st.subheader("Input Images")
    imgs = st.file_uploader("Choose Images", accept_multiple_files=True)

    color_data_list = []  # 用于保存每张饼图的数据

    if imgs is not None:
        for i, img in enumerate(imgs):
            st.subheader("Image " + str(i + 1))
            st.image(img)

            img_data = np.array(Image.open(img))

            if img_data.shape[2] == 4:  # Check if image has 4 channels (alpha channel)
                alpha_channel = img_data[:, :, 3]
                alpha_mask = alpha_channel != 0
                img_data = img_data[alpha_mask]
                img_data = img_data[:, :3]  # Remove alpha channel if present

            data = img_data.reshape((-1, 3))
            data = np.float32(data)

            # 使用肘部法则选择最佳的k值
            sse = []
            k_values = range(2, 13)  # 尝试的k值范围
            # 清空SSE列表
            sse.clear()
            for k in k_values:
                criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
                ret, label, center = cv2.kmeans(data, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
                sse.append(np.sum((data - center[label.flatten()]) ** 2))
            
            
            plt.clf() 
            plt.plot(k_values, sse, 'bx-')
            plt.xlabel('Number of clusters (k)')
            plt.ylabel('Sum of Squared Errors (SSE)')
            plt.title('Elbow Method for Optimal k')
            st.pyplot(plt)  # 使用st.pyplot显示曲线图

            # 自动选择最佳的k值
            diff = np.diff(sse)
            diff_ratio = diff[:-1] / diff[1:]
            optimal_k = np.argmax(diff_ratio) + 2  # k值的索引需要加2，因为diff_ratio比diff少一个元素
            k = int(optimal_k)

            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
            ret, label, center = cv2.kmeans(data, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

            unique, counts = np.unique(label, return_counts=True)
            ratios = counts / len(data)

            colors = center.astype(int)
            labels = np.arange(k)

            fig, ax = plt.subplots()  # 创建新的图形对象
            ax.pie(ratios, labels=labels, colors=colors / 255, autopct='%1.1f%%', startangle=90)
            ax.axis('equal')
            ax.set_title('Extracted Colors')

            color_data_list.append({"Image": img, "Colors": colors, "Ratios": ratios})
            #colors_ratios = np.concatenate((colors, np.expand_dims(ratios, axis=1)), axis=1)
            #df = pd.DataFrame(colors_ratios)

            st.write("Colors and Ratios for Image " + str(i + 1))
            #st.write(df)  # 显示数据帧，并设置标题行
            st.pyplot(fig)
    
    return color_data_list


def total_pie_chart(color_data_list):
    total_img = []
    total_colors = []
    k_total = st.number_input('Enter the no. of colors to be extracted in the Total Pie Chart', min_value=1)
    if len(color_data_list) > 0:       
        for color_data in color_data_list:
            colors = color_data["Colors"]
            ratios = color_data["Ratios"]
            iterations = [int(ratio * 100) for ratio in ratios]
            for color, iteration in zip(colors, iterations):
                total_img.extend([color] * iteration)

        total_data = np.float32(total_img)

        # 使用KMeans算法进行聚类
        model = KMeans(n_clusters=k_total)
        model.fit(total_data)

        labels = model.labels_
        centers = model.cluster_centers_

        unique, counts = np.unique(labels, return_counts=True)
        ratios = counts / len(total_data)

        colors = centers.astype(int)
        labels = np.arange(k_total)

        fig_total, ax_total = plt.subplots()
        ax_total.pie(ratios, labels=labels, colors=colors / 255, autopct='%1.1f%%', startangle=90)
        ax_total.axis('equal')
        ax_total.set_title('Total Extracted Colors')

        total_colors.append({"Colors": colors, "Ratios": ratios})
        colors_ratios = np.concatenate((colors, np.expand_dims(ratios, axis=1)), axis=1)
        df = pd.DataFrame(colors_ratios)

        st.write("Colors and Ratios for Total Pie Chart")
        st.write(df)  
        st.pyplot(fig_total)
    return total_colors

def color_network_model(color_data_list, total_colors):
    t_value = st.number_input('Enter the Threshold Value', value=0.1)
    total_marked_colors = []
    marked_colors = []
    G = nx.Graph()
    if len(total_colors) > 0:   
        for i, color_data in enumerate(total_colors):
            colors = color_data["Colors"]
            ratios = color_data["Ratios"]
            for color, ratio in zip(colors, ratios):
                if ratio > 0:
                    G.add_node(tuple(color), size=ratio)

        for img_color_data in color_data_list:
            img_colors = img_color_data["Colors"]
            for img_color in img_colors:
                for color_data in total_colors:
                    colors = color_data["Colors"]
                    for color in colors:
                        if (
                            color[0] * (1 - t_value) <= img_color[0] <= color[0] * (1 + t_value)
                            and color[1] * (1 - t_value) <= img_color[1] <= color[1] * (1 + t_value)
                            and color[2] * (1 - t_value) <= img_color[2] <= color[2] * (1 + t_value)
                        ):
                            marked_colors.append(tuple(color.tolist()))
            total_marked_colors.append(marked_colors)
            marked_colors = []  

        plt.figure(figsize=(10, 10))
        pos = nx.circular_layout(G)

        for color_array in total_marked_colors:
            for i in range(len(color_array)):
                for j in range(i + 1, len(color_array)):
                    if G.has_edge(color_array[i], color_array[j]):
                        # 如果已连接，则增加连线的宽度
                        G[color_array[i]][color_array[j]]["width"] += 0.5
                    else:
                        # 如果没有连接，则添加连线
                        G.add_edge(color_array[i], color_array[j], width=0.5)

        # 绘制图形
        node_sizes = [G.nodes[n]["size"] * 10000 for n in G.nodes()]
        node_color = [tuple(c / 255 for c in color) for color in list(G.nodes())]
        edge_widths = [G[u][v]["width"] for u, v in G.edges()]
        nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color=node_color)
        nx.draw_networkx_edges(G, pos, width=edge_widths)
        plt.axis("off")
        plt.title('Color Network Model')
        st.pyplot(plt.gcf())
        print(total_marked_colors)
        return total_marked_colors, G
    return [], nx.Graph()

def color_application(total_marked_colors, G):
    st.subheader("Select Colors And Apply Them")
    input_value = st.text_input("Enter the rgb value of the main color and separate it with a space.")
    values = tuple(int(x) for x in input_value.split(" ") if x.strip())
    st.write("Entered value:", values)
    k = st.number_input('Enter the number of auxiliary colors:', min_value=1)       
    
    if len(total_marked_colors) > 0:     

        connected_edges = []
        selected_colors = [values]
        for u, v in G.edges():
            if u == values or v == values:
                connected_edges.append((u, v))

        if len(connected_edges) > 0:
            # 根据边的宽度排序
            sorted_edges = sorted(connected_edges, key=lambda edge: G[edge[0]][edge[1]]["width"], reverse=True)
            # 取前 k 个边
            top_k_edges = sorted_edges[:k]

            # 输出结果
            st.subheader(f"The most correlated {k} colors:")
            for idx, edge in enumerate(top_k_edges):
                color1 = edge[0]
                color2 = edge[1]
                if color1 == values:
                    st.write(f"Color{idx+1}:{color2}, Width:{G[color1][color2]['width']}")
                    selected_colors.append(color2)
                else:
                    st.write(f"Color{idx+1}:{color1}, Width:{G[color1][color2]['width']}")
                    selected_colors.append(color1)
        else:
            st.write("The edge associated with the input color cannot be found.")
        
        st.subheader("Input Images")
        original_img = st.file_uploader("Choose Images")
        print(selected_colors)

        if original_img is not None:
            # 将选定的颜色转换为numpy数组
            selected_colors_np = np.array(selected_colors)

            # 读取原始图像
            img = Image.open(original_img)
            img_array = np.array(img)

            # 创建新图像数组
            new_img_array = np.zeros_like(img_array)
            if new_img_array.shape[2] == 4:  # Check if image has 4 channels
                new_img_array = new_img_array[:, :, :3]  # Remove alpha channel if present
            # 遍历原始图像的像素
            for i in range(img_array.shape[0]):
                for j in range(img_array.shape[1]):
                    # 计算原始像素颜色与选定颜色的距离
                    distances = np.sqrt(np.sum((selected_colors_np - img_array[i, j])**2, axis=1))
                    # 找到最近的颜色索引
                    closest_color_idx = np.argmin(distances)
                    # 将最近的颜色赋给新图像的对应像素
                    new_img_array[i, j] = selected_colors_np[closest_color_idx]

            # 创建新图像对象并显示
            new_img = Image.fromarray(new_img_array)
            st.subheader("New Image")
            st.image(new_img)

def main():
    with st.sidebar:
 
        if st.session_state.get('switch_button', False):
            st.session_state['menu_option'] = (st.session_state.get('menu_option', 0) + 1) % 4
            manual_select = st.session_state['menu_option']
        else:
            manual_select = None      
        selected4 = option_menu("Main Menu", ["Extracted Colors", "Total Pie Chart", "Color Network Model", 'Color Application'], 
                                icons=['house', 'cloud-upload', "list-task", 'gear'], menu_icon="cast", 
                                orientation="vertical", manual_select=manual_select, key='menu_4')
        st.button(f"Move to Next {st.session_state.get('menu_option', 1)}", key='switch_button')
    st.title(selected4)

    color_data_list = st.session_state.get("color_data_list", [])
    total_colors = st.session_state.get("total_colors", [])
    total_marked_colors = st.session_state.get("total_marked_colors", [])
    G = st.session_state.get("G")
    if selected4 == "Extracted Colors":
        color_data_list = extract_colors()
        st.session_state["color_data_list"] = color_data_list
    if selected4 == "Total Pie Chart":
        total_colors = total_pie_chart(color_data_list, )
        st.session_state["total_colors"] = total_colors
    if selected4 == "Color Network Model":
        total_marked_colors, G = color_network_model(color_data_list, total_colors)
        st.session_state["total_marked_colors"] = total_marked_colors
        st.session_state["G"] = G
    if selected4 == "Color Application":
        color_application(total_marked_colors, G)

main()


