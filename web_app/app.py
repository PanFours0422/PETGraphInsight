import warnings
warnings.filterwarnings("ignore")

from flask import Flask, render_template, jsonify, request
import sys
import os

# 添加项目根目录到Python路径
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

from data.generate_data import generate_detector_data, generate_lm_data
from graph.build_graph import build_graph, save_graph
from graph.build_subgraphs import build_training_samples, save_samples
from ml.predict import load_model, predict_coupling
import pandas as pd
import torch
import numpy as np

app = Flask(__name__)

@app.route('/')
def index():
    """渲染主页"""
    return render_template('index.html')

@app.route('/api/generate_data', methods=['POST'])
def generate_data():
    """生成数据API"""
    try:
        data = request.get_json()
        num_detectors = data.get('num_detectors', 100)
        num_events = data.get('num_events', 1000)
        
        # 生成数据
        detector_data = generate_detector_data(num_detectors)
        lm_data = generate_lm_data(num_events, detector_data)
        
        # 保存数据到项目根目录的data/raw下
        raw_data_dir = os.path.join(ROOT_DIR, 'data', 'raw')
        os.makedirs(raw_data_dir, exist_ok=True)
        
        detector_data.to_csv(os.path.join(raw_data_dir, 'detector.csv'), index=False)
        lm_data.to_csv(os.path.join(raw_data_dir, 'lm_data.csv'), index=False)
        
        return jsonify({
            'status': 'success',
            'message': f'成功生成{num_detectors}个探测器和{len(lm_data)}个事件'
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        })

@app.route('/api/build_graph', methods=['POST'])
def build_pet_graph():
    """构建PET图API"""
    try:
        # 构建主图
        raw_data_dir = os.path.join(ROOT_DIR, 'data', 'raw')
        detector_data = pd.read_csv(os.path.join(raw_data_dir, 'detector.csv'))
        lm_data = pd.read_csv(os.path.join(raw_data_dir, 'lm_data.csv'))
        
        # 构建图
        graph = build_graph(detector_data, lm_data)
        
        # 保存图
        processed_dir = os.path.join(ROOT_DIR, 'data', 'processed')
        os.makedirs(processed_dir, exist_ok=True)
        save_graph(graph, processed_dir)
        
        # 构建子图
        subgraphs, labels = build_training_samples(graph)
        save_samples(subgraphs, labels, processed_dir)
        
        return jsonify({
            'status': 'success',
            'message': f'成功构建PET图，包含{graph.num_nodes}个节点和{graph.edge_index.shape[1]}条边，生成{len(subgraphs)}个子图'
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        })

@app.route('/api/preview_data')
def preview_data():
    """预览数据API"""
    try:
        preview_type = request.args.get('type', 'detectors')
        limit = int(request.args.get('limit', 100))
        
        raw_data_dir = os.path.join(ROOT_DIR, 'data', 'raw')
        
        if preview_type == 'detectors':
            detector_data = pd.read_csv(os.path.join(raw_data_dir, 'detector.csv'))
            positions = detector_data[['x', 'y', 'z']].values[:limit].tolist()
            return jsonify({
                'status': 'success',
                'data': {
                    'type': 'detectors',
                    'positions': positions
                }
            })
        elif preview_type == 'events':
            lm_data = pd.read_csv(os.path.join(raw_data_dir, 'lm_data.csv'))
            events = lm_data.head(limit).to_dict('records')
            return jsonify({
                'status': 'success',
                'data': {
                    'type': 'events',
                    'events': events
                }
            })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        })

@app.route('/api/predict', methods=['POST'])
def predict():
    """预测耦合度API"""
    try:
        # 设置设备
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 加载数据
        processed_dir = os.path.join(ROOT_DIR, 'data', 'processed')
        test_subgraphs = torch.load(os.path.join(processed_dir, "subgraphs.pt"), weights_only=False)
        dataset_info = torch.load(os.path.join(processed_dir, "dataset_info.pt"), weights_only=False)
        
        # 加载模型
        model_dir = os.path.join(ROOT_DIR, "models")
        model_path = os.path.join(model_dir, "best_model.pt")
        
        if not os.path.exists(model_path):
            return jsonify({
                'status': 'error',
                'message': '找不到模型文件，请先训练模型'
            })
        
        model = load_model(
            model_path,
            in_channels=dataset_info['node_feature_dim'],
            hidden_channels=64,
            out_channels=2
        ).to(device)
        
        # 进行预测
        predictions, probabilities = predict_coupling(model, test_subgraphs, device)
        
        # 保存预测结果
        results_dir = os.path.join(ROOT_DIR, "results")
        os.makedirs(results_dir, exist_ok=True)
        
        # 保存预测结果
        results = {
            'predictions': predictions,  # 预测的类别（0或1）
            'probabilities': probabilities,  # 预测的概率分布
            'coupling_level': ['低耦合' if p == 0 else '高耦合' for p in predictions]  # 耦合度描述
        }
        
        # 保存为numpy文件
        np.save(os.path.join(results_dir, 'prediction_results.npy'), results)
        
        # 保存详细的预测结果到文本文件
        with open(os.path.join(results_dir, 'prediction_details.txt'), 'w', encoding='utf-8') as f:
            f.write("预测结果详细信息:\n")
            f.write("-" * 50 + "\n")
            for i, (pred, prob) in enumerate(zip(predictions, probabilities)):
                f.write(f"样本 {i+1}:\n")
                f.write(f"预测类别: {'高耦合' if pred == 1 else '低耦合'}\n")
                f.write(f"低耦合概率: {prob[0]:.4f}\n")
                f.write(f"高耦合概率: {prob[1]:.4f}\n")
                f.write("-" * 30 + "\n")
        
        return jsonify({
            'status': 'success',
            'message': f'预测完成，共预测{len(predictions)}个样本，其中高耦合区域{np.sum(predictions == 1)}个，低耦合区域{np.sum(predictions == 0)}个'
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        })

@app.route('/api/preview_prediction')
def preview_prediction():
    """预览预测结果API"""
    try:
        results_dir = os.path.join(ROOT_DIR, "results")
        results_path = os.path.join(results_dir, 'prediction_results.npy')
        
        if not os.path.exists(results_path):
            return jsonify({
                'status': 'error',
                'message': '找不到预测结果文件'
            })
        
        # 加载预测结果
        results = np.load(results_path, allow_pickle=True).item()
        
        # 获取筛选类型
        filter_type = request.args.get('type', 'all')
        
        # 准备预览数据
        preview_data = []
        for i in range(len(results['predictions'])):
            # 根据筛选类型过滤数据
            if filter_type == 'all' or \
               (filter_type == 'high' and results['predictions'][i] == 1) or \
               (filter_type == 'low' and results['predictions'][i] == 0):
                preview_data.append({
                    'coupling_level': results['coupling_level'][i],
                    'probabilities': results['probabilities'][i].tolist()  # 转换为Python列表以便JSON序列化
                })
        
        return jsonify({
            'status': 'success',
            'data': {
                'predictions': preview_data[:100]  # 只返回前100个结果用于预览
            }
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        })

@app.route('/api/get_low_coupling_pairs')
def get_low_coupling_pairs():
    """获取低耦合区域的探测器对信息API"""
    try:
        # 加载预测结果
        results_dir = os.path.join(ROOT_DIR, "results")
        results_path = os.path.join(results_dir, 'prediction_results.npy')
        
        if not os.path.exists(results_path):
            return jsonify({
                'status': 'error',
                'message': '找不到预测结果文件'
            })
        
        # 加载预测结果和原始数据
        results = np.load(results_path, allow_pickle=True).item()
        processed_dir = os.path.join(ROOT_DIR, 'data', 'processed')
        test_subgraphs = torch.load(os.path.join(processed_dir, "subgraphs.pt"), weights_only=False)
        
        # 获取低耦合样本的索引
        low_coupling_indices = np.where(results['predictions'] == 0)[0]
        
        # 加载原始事件数据
        raw_data_dir = os.path.join(ROOT_DIR, 'data', 'raw')
        lm_data = pd.read_csv(os.path.join(raw_data_dir, 'lm_data.csv'))
        
        # 存储低耦合区域的探测器对信息
        low_coupling_pairs = []
        
        for idx in low_coupling_indices:
            # 获取子图中的节点对
            subgraph = test_subgraphs[idx]
            node_pairs = subgraph.edge_index.t().numpy()
            
            # 获取每对探测器的坐标
            for pair in node_pairs:
                detector_i, detector_j = pair[0], pair[1]
                
                # 从lm_data中找到对应的探测器对事件
                event_data = lm_data[
                    (lm_data['detector_i'] == detector_i) & 
                    (lm_data['detector_j'] == detector_j)
                ].iloc[0] if len(lm_data[
                    (lm_data['detector_i'] == detector_i) & 
                    (lm_data['detector_j'] == detector_j)
                ]) > 0 else None
                
                if event_data is not None:
                    # 计算中点坐标
                    midpoint = [
                        (float(event_data['pos_i_x']) + float(event_data['pos_j_x'])) / 2,
                        (float(event_data['pos_i_y']) + float(event_data['pos_j_y'])) / 2,
                        (float(event_data['pos_i_z']) + float(event_data['pos_j_z'])) / 2
                    ]
                    
                    # 获取探测器对的详细信息
                    detector_pair_info = {
                        'midpoint': midpoint,
                        'coupling_probability': float(results['probabilities'][idx][0]),
                        'detector_pair': [int(detector_i), int(detector_j)],
                        'detector_i_pos': [
                            float(event_data['pos_i_x']),
                            float(event_data['pos_i_y']),
                            float(event_data['pos_i_z'])
                        ],
                        'detector_j_pos': [
                            float(event_data['pos_j_x']),
                            float(event_data['pos_j_y']),
                            float(event_data['pos_j_z'])
                        ],
                        'energy_i': float(event_data['energy_i']),
                        'energy_j': float(event_data['energy_j']),
                        'timestamp_i': float(event_data['timestamp_i']),
                        'timestamp_j': float(event_data['timestamp_j'])
                    }
                    
                    low_coupling_pairs.append(detector_pair_info)
        
        return jsonify({
            'status': 'success',
            'data': {
                'low_coupling_pairs': low_coupling_pairs
            }
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        })

if __name__ == '__main__':
    app.run(debug=True, port=5000)
