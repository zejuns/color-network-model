import streamlit as st
import numpy as np
import cv2
from matplotlib import pyplot as plt
from PIL import Image
import pandas as pd
import networkx as nx
from sklearn.cluster import KMeans
from streamlit_option_menu import option_menu
import zipfile

@st.cache_data
def extract_colors(imgs, max_k):
    if imgs is not None:
        color_data_list = []  # 用于保存每张饼图的数据
        for i, img in enumerate(imgs):
            st.subheader("Image " + str(i + 1))
            col1, col2, col3 = st.columns(3)
            with col1:
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
            k_values = range(2, max_k + 1)  # 尝试的k值范围
            for k in k_values:
                criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
                ret, label, center = cv2.kmeans(data, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
                sse.append(np.sum((data - center[label.flatten()]) ** 2))
            
            plt.clf() 
            plt.plot(k_values, sse, 'bx-')
            plt.xlabel('Number of clusters (k)')
            plt.ylabel('Sum of Squared Errors (SSE)')
            plt.title('Elbow Method for Optimal k')
            plt.plot([k_values[0], k_values[-1]], [sse[0], sse[-1]], 'r-')  # 添加连接第一个点和最后一个点的红色直线
            with col2:
                st.pyplot(plt)  # 使用st.pyplot显示曲线图

            # 自动选择最佳的k值
            line_slope = (sse[-1] - sse[0]) / (k_values[-1] - k_values[0])  # 直线a的斜率
            m_values = line_slope * np.array(k_values) - np.array(sse)  # 计算m值
            optimal_k = k_values[np.argmax(m_values)]  # 获取最大m值对应的k值
            k = int(optimal_k)

            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
            ret, label, center = cv2.kmeans(data, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

            unique, counts = np.unique(label, return_counts=True)
            ratios = counts / len(data)

            colors = center.astype(int)
            labels = np.arange(1, k+1)

            fig, ax = plt.subplots()  # 创建新的图形对象
            sorted_indices = np.argsort(ratios)[::-1]  # 按比例从大到小排序的索引
            ratios = ratios[sorted_indices]
            colors = colors[sorted_indices]
            ax.pie(ratios, labels=labels, colors=colors / 255, autopct='%1.1f%%', startangle=90)
            ax.axis('equal')
            ax.set_title('Extracted Colors')

            color_data_list.append({"Image": img, "Colors": colors, "Ratios": ratios})
            with col3:
                st.pyplot(fig)
    return color_data_list


def download_images(color_data_list):
    if len(color_data_list) > 0:
        # 保存饼图和颜色数据
        for i, color_data in enumerate(color_data_list):
            img = color_data["Image"]
            colors = color_data["Colors"]
            ratios = color_data["Ratios"]

            fig, ax = plt.subplots(figsize=(8, 8))  # 创建新的图形对象
            ax.pie(ratios, labels=None, colors=colors / 255, autopct='', startangle=90)
            ax.axis('equal')
            ax.set_title('')

            # 保存饼图为png格式并设定背景透明
            fig.patch.set_alpha(0)
            fig.savefig(f"pie_{i+1}.png", transparent=True)
            plt.close(fig)

        # 创建一个下载按钮
        with open("pies.zip", "wb") as file:
            with zipfile.ZipFile(file, "w") as zipf:
                for i in range(len(color_data_list)):
                    zipf.write(f"pie_{i+1}.png")
    
        # 显示下载按钮
        with open("pies.zip", "rb") as file:
            btn = st.download_button(
                label="Download All Pie Charts",
                data=file,
                file_name="pies.zip",
                mime="application/zip",
                use_container_width=True,
            )


def total_pie_chart(color_data_list, k_total):
    col1, col2 = st.columns(2)           
    total_img = []
    total_colors = []
    
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
        labels = np.arange(1, k_total+1)

        fig_total, ax_total = plt.subplots()
        sorted_indices = np.argsort(ratios)[::-1]  # 按比例从大到小排序的索引
        ratios = ratios[sorted_indices]
        colors = colors[sorted_indices]
        ax_total.pie(ratios, labels=labels, colors=colors / 255, autopct='', startangle=90)
        ax_total.axis('equal')
        ax_total.set_title('Total Pie Chart')
        # 调整标题和饼图之间的距离
        plt.subplots_adjust(top=1.2)

        total_colors.append({"Colors": colors, "Ratios": ratios})
        colors_ratios = np.concatenate((colors, np.expand_dims(ratios, axis=1)), axis=1)
        df = pd.DataFrame(colors_ratios, columns=["R", "G", "B", "Ratio"])
        df.index = df.index + 1
        with col1:
            st.pyplot(fig_total)
        with col2:
            st.write(df)  
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
        node_sizes = [G.nodes[n]["size"] * 6000 for n in G.nodes()]
        node_color = [tuple(c / 255 for c in color) for color in list(G.nodes())]
        edge_widths = [G[u][v]["width"] for u, v in G.edges()]

        # 调整节点坐标，将节点以中心点为中心放大一圈
        scaled_pos = {k: (v[0]*1.1, v[1]*1.1) for k, v in pos.items()}
        
        nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color=node_color)
        nx.draw_networkx_edges(G, pos, width=edge_widths)
        nx.draw_networkx_labels(G, scaled_pos, font_size=8, font_color='black', labels={n: str(i+1) for i, n in enumerate(G.nodes())})
        plt.axis("off")
        plt.title('Color Network Model')
        st.pyplot(plt.gcf())
        print(total_marked_colors)
        return total_marked_colors, G
    return [], nx.Graph()

def color_application(total_marked_colors, G):
    st.subheader("Select Colors And Apply Them")
    input_value = st.text_input("Enter the No. of the Main Color in the Total Pie Chart", value=1)
    values = ()
    counter = 0

    for node in G.nodes:
        if counter == int(input_value) - 1:
            rgb_value = node
            values = rgb_value
            break       
        counter += 1
    else:
        values = []  # 如果输入值为空或超出节点范围，则将 values 设置为空列表
    st.write("RGB value:", values)
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
            if img_array.shape[2] == 4:  # Check if image has 4 channels (alpha channel)
                img_array = img_array[:, :, :3]
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
        manual_select = None      
        selected4 = option_menu("Main Menu", ["Extracted Colors", "Total Pie Chart", "Color Network Model", 'Color Application'], 
                                icons=['house', 'cloud-upload', "list-task", 'gear'], menu_icon="cast", 
                                orientation="vertical", manual_select=manual_select, key='menu_4')
    st.title(selected4)
    
    imgs = st.session_state.get("imgs", [])
    color_data_list = st.session_state.get("color_data_list", [])
    total_colors = st.session_state.get("total_colors", [])
    total_marked_colors = st.session_state.get("total_marked_colors", [])
    G = st.session_state.get("G")
    if selected4 == "Extracted Colors":
        st.subheader("Input Images")
        max_k = st.number_input('Enter the Max K of K-means', value=20)        
        imgs = st.file_uploader("Choose Images", accept_multiple_files=True)
        color_data_list = extract_colors(imgs, max_k)
        download_images(color_data_list)
        st.session_state["imgs"] = imgs
        st.session_state["color_data_list"] = color_data_list
    if selected4 == "Total Pie Chart":
        k_total = st.number_input('Enter the No. of Colors to Be Extracted', min_value=1)
        total_colors = total_pie_chart(color_data_list, k_total)
        st.session_state["total_colors"] = total_colors
    if selected4 == "Color Network Model":
        total_marked_colors, G = color_network_model(color_data_list, total_colors)
        st.session_state["total_marked_colors"] = total_marked_colors
        st.session_state["G"] = G
    if selected4 == "Color Application":
        color_application(total_marked_colors, G)

main()


