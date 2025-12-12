"""
定价不经意传输协议(POT)改进方案复现代码
作者：齐轲 (20233001410)
时间：2025年12月

本代码实现了论文中描述的改进POT协议，包含：
1. NTRU后量子加密算法
2. 差分隐私保护机制（拉普拉斯噪声）
3. 博弈论定价模型
4. 完整的协议流程
"""

import numpy as np
import random
import hashlib
import json
import time
from typing import Tuple, List, Dict, Any
from dataclasses import dataclass
from scipy.stats import laplace
import matplotlib.pyplot as plt
from collections import defaultdict


@dataclass
class NTRUParameters:
    """NTRU算法参数配置"""
    N: int = 503  # 多项式次数
    p: int = 3  # 小模数
    q: int = 256  # 大模数
    d: int = 251  # 多项式f,g中系数为1的个数


@dataclass
class PrivacyParameters:
    """差分隐私参数配置"""
    epsilon: float = 0.1  # 隐私预算
    sensitivity: float = 1.0  # 灵敏度
    delta: float = 1e-5  # (ε,δ)-差分隐私参数


@dataclass
class GameTheoryParameters:
    """博弈论定价参数配置"""
    k: float = 0.1  # 需求敏感度
    alpha: float = 0.05  # 供给敏感度
    learning_rate: float = 0.1  # 学习率
    max_iterations: int = 100  # 最大迭代次数
    tolerance: float = 0.01  # 收敛容差


class NTruCryptosystem:
    """NTRU后量子加密系统实现"""

    def __init__(self, params: NTRUParameters):
        self.params = params
        self.N = params.N
        self.p = params.p
        self.q = params.q
        self.d = params.d

    def generate_small_poly(self) -> np.ndarray:
        """生成小系数多项式（系数为-1,0,1）"""
        poly = np.zeros(self.N, dtype=int)
        # 随机选择d个位置设为1
        ones_pos = random.sample(range(self.N), self.d)
        for pos in ones_pos:
            poly[pos] = 1
        # 随机选择d个位置设为-1
        neg_ones_pos = random.sample([i for i in range(self.N) if i not in ones_pos], self.d)
        for pos in neg_ones_pos:
            poly[pos] = -1
        return poly

    def polynomial_mod(self, poly: np.ndarray, modulus: int) -> np.ndarray:
        """多项式模运算"""
        return np.mod(poly, modulus)

    def polynomial_convolution(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """多项式卷积计算（循环卷积）"""
        N = len(a)
        result = np.zeros(N, dtype=int)
        for i in range(N):
            for j in range(N):
                result[(i + j) % N] += a[i] * b[j]
        return result

    def find_inverse(self, f: np.ndarray, modulus: int) -> np.ndarray:
        """计算多项式在模modulus下的逆"""
        # 使用扩展欧几里得算法求逆
        N = self.N
        # 构建伴随矩阵
        mat = np.zeros((N, N), dtype=int)
        for i in range(N):
            for j in range(N):
                mat[i][j] = f[(j - i) % N]

        # 计算行列式（简化实现）
        det = 1
        for i in range(N):
            det = (det * mat[i][i]) % modulus

        if det == 0:
            raise ValueError("多项式不可逆")

        # 返回简化逆（实际应用应使用完整求逆算法）
        inverse = np.zeros(N, dtype=int)
        for i in range(N):
            if f[i] != 0:
                inverse[i] = pow(int(f[i]), -1, modulus)
        return inverse

    def generate_key_pair(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """生成NTRU密钥对"""
        # 生成私钥f
        while True:
            f = self.generate_small_poly()
            try:
                f_p = self.find_inverse(f, self.p)
                f_q = self.find_inverse(f, self.q)
                break
            except ValueError:
                continue

        # 生成多项式g
        g = self.generate_small_poly()

        # 计算公钥h = p * f_q * g mod q
        fq_times_g = self.polynomial_convolution(f_q, g)
        p_times_fqg = (self.p * fq_times_g) % self.q
        h = self.polynomial_mod(p_times_fqg, self.q)

        return h, f, f_p

    def encrypt(self, message: str, public_key: np.ndarray) -> np.ndarray:
        """NTRU加密"""
        # 编码消息为多项式
        message_bytes = message.encode('utf-8')
        message_poly = self.encode_message(message_bytes)

        # 生成随机多项式r
        r = self.generate_small_poly()

        # 计算密文: e = r * h + m mod q
        r_times_h = self.polynomial_convolution(r, public_key)
        e = self.polynomial_mod(r_times_h + message_poly, self.q)

        return e

    def decrypt(self, ciphertext: np.ndarray, private_key_f: np.ndarray,
                private_key_fp: np.ndarray) -> str:
        """NTRU解密"""
        # 计算: a = f * e mod q
        a = self.polynomial_convolution(private_key_f, ciphertext)
        a = self.polynomial_mod(a, self.q)

        # 中心化系数到[-q/2, q/2]
        a_centered = np.array([(x + self.q // 2) % self.q - self.q // 2 for x in a])

        # 计算: m = fp * a mod p
        m = self.polynomial_convolution(private_key_fp, a_centered)
        m = self.polynomial_mod(m, self.p)

        # 解码多项式为消息
        message_bytes = self.decode_message(m)

        return message_bytes.decode('utf-8', errors='ignore')

    def encode_message(self, message_bytes: bytes) -> np.ndarray:
        """编码消息为多项式"""
        poly = np.zeros(self.N, dtype=int)
        bit_length = min(len(message_bytes) * 8, self.N)

        for i in range(bit_length):
            byte_idx = i // 8
            bit_idx = i % 8
            if byte_idx < len(message_bytes):
                bit = (message_bytes[byte_idx] >> bit_idx) & 1
                poly[i] = bit

        return poly

    def decode_message(self, poly: np.ndarray) -> bytes:
        """解码多项式为消息"""
        bytes_list = []
        current_byte = 0
        bit_count = 0

        for i, coeff in enumerate(poly):
            bit = 1 if coeff == 1 else 0
            current_byte |= (bit << bit_count)
            bit_count += 1

            if bit_count == 8:
                bytes_list.append(current_byte)
                current_byte = 0
                bit_count = 0

        return bytes(bytes_list)


class DifferentialPrivacyEngine:
    """差分隐私引擎实现"""

    def __init__(self, params: PrivacyParameters):
        self.params = params
        self.epsilon_total = params.epsilon
        self.epsilon_used = 0.0
        self.delta = params.delta
        self.sensitivity = params.sensitivity

    def laplace_mechanism(self, true_value: float, epsilon: float = None) -> Tuple[float, Dict]:
        """拉普拉斯机制实现ε-差分隐私"""
        if epsilon is None:
            epsilon = self.epsilon_total - self.epsilon_used

        # 计算尺度参数
        scale = self.sensitivity / epsilon

        # 生成拉普拉斯噪声
        noise = np.random.laplace(0, scale)

        # 计算含噪值
        noisy_value = true_value + noise

        # 更新隐私预算
        self.epsilon_used += epsilon

        # 记录隐私消耗
        privacy_log = {
            'true_value': true_value,
            'noisy_value': noisy_value,
            'noise': noise,
            'epsilon_used': epsilon,
            'scale': scale,
            'mechanism': 'laplace'
        }

        return noisy_value, privacy_log

    def gaussian_mechanism(self, true_value: float, epsilon: float = None) -> Tuple[float, Dict]:
        """高斯机制实现(ε,δ)-差分隐私"""
        if epsilon is None:
            epsilon = self.epsilon_total - self.epsilon_used

        # 计算标准差
        sigma = self.sensitivity * np.sqrt(2 * np.log(1.25 / self.delta)) / epsilon

        # 生成高斯噪声
        noise = np.random.normal(0, sigma)

        # 计算含噪值
        noisy_value = true_value + noise

        # 更新隐私预算
        self.epsilon_used += epsilon

        # 记录隐私消耗
        privacy_log = {
            'true_value': true_value,
            'noisy_value': noisy_value,
            'noise': noise,
            'epsilon_used': epsilon,
            'sigma': sigma,
            'delta': self.delta,
            'mechanism': 'gaussian'
        }

        return noisy_value, privacy_log

    def exponential_mechanism(self, candidates: List[Any],
                              scores: List[float],
                              sensitivity: float = None) -> Tuple[Any, float]:
        """指数机制实现"""
        if sensitivity is None:
            sensitivity = self.sensitivity

        # 归一化分数
        max_score = max(scores)
        normalized_scores = [score - max_score for score in scores]

        # 计算选择概率
        probabilities = [np.exp(self.params.epsilon * score / (2 * sensitivity))
                         for score in normalized_scores]
        total_prob = sum(probabilities)
        probabilities = [p / total_prob for p in probabilities]

        # 根据概率选择
        chosen_idx = np.random.choice(len(candidates), p=probabilities)

        return candidates[chosen_idx], probabilities[chosen_idx]

    def calculate_sensitivity(self, query_type: str, data_range: Tuple) -> float:
        """计算查询灵敏度"""
        if query_type == 'count':
            return 1.0
        elif query_type == 'sum':
            return data_range[1] - data_range[0]
        elif query_type == 'mean':
            return (data_range[1] - data_range[0]) / data_range[2] if len(data_range) > 2 else 1.0
        elif query_type == 'price':
            # 价格查询灵敏度
            return min(data_range[1] * 0.1, 10.0)
        else:
            return 1.0


class GameTheoryPricing:
    """博弈论定价引擎实现"""

    def __init__(self, params: GameTheoryParameters):
        self.params = params
        self.price_history = []
        self.equilibrium_prices = {}

    def demand_function(self, price: float, value: float, k: float = None) -> float:
        """需求函数（sigmoid形式）"""
        if k is None:
            k = self.params.k
        return 1.0 / (1.0 + np.exp(k * (price - value)))

    def supply_function(self, price: float, cost: float, alpha: float = None) -> float:
        """供给函数（指数形式）"""
        if alpha is None:
            alpha = self.params.alpha
        return np.exp(-alpha * (price - cost))

    def calculate_equilibrium_price(self, item_id: int, value: float, cost: float) -> Tuple[float, int]:
        """计算单个商品的均衡价格"""
        p_current = (value + cost) / 2.0
        iterations = 0

        for i in range(self.params.max_iterations):
            iterations = i + 1

            # 计算需求和供给
            demand = self.demand_function(p_current, value)
            supply = self.supply_function(p_current, cost)

            # 计算价格调整
            p_new = p_current + self.params.learning_rate * (demand * (p_current - cost) - supply * cost)

            # 确保价格合理
            p_new = max(p_new, cost * 1.1)  # 不低于成本110%
            p_new = min(p_new, value * 0.9)  # 不高于估值90%

            # 检查收敛
            if abs(p_new - p_current) < self.params.tolerance:
                p_current = p_new
                break

            p_current = p_new

        # 记录均衡价格
        self.equilibrium_prices[item_id] = p_current
        self.price_history.append({
            'item_id': item_id,
            'equilibrium_price': p_current,
            'iterations': iterations,
            'value': value,
            'cost': cost
        })

        return p_current, iterations

    def calculate_nash_equilibrium(self, items: List[Tuple[int, float, float]]) -> Dict[int, float]:
        """计算多商品纳什均衡"""
        nash_prices = {}

        # 初步计算每个商品的均衡价格（不考虑竞争）
        preliminary_prices = {}
        for item_id, value, cost in items:
            eq_price, _ = self.calculate_equilibrium_price(item_id, value, cost)
            preliminary_prices[item_id] = eq_price

        # 考虑竞争调整价格
        for item_id, value, cost in items:
            # 计算竞争因子
            competition_factor = self.calculate_competition_factor(item_id, items, preliminary_prices)

            # 调整估值考虑竞争
            adjusted_value = value * competition_factor

            # 重新计算均衡价格
            eq_price, _ = self.calculate_equilibrium_price(item_id, adjusted_value, cost)
            nash_prices[item_id] = eq_price

        return nash_prices

    def calculate_competition_factor(self, target_item_id: int,
                                     items: List[Tuple[int, float, float]],
                                     prices: Dict[int, float]) -> float:
        """计算竞争因子"""
        target_value = None
        total_alternative_value = 0
        alternative_count = 0

        for item_id, value, _ in items:
            if item_id == target_item_id:
                target_value = value
            else:
                total_alternative_value += value
                alternative_count += 1

        if target_value is None or alternative_count == 0:
            return 1.0

        avg_alternative_value = total_alternative_value / alternative_count

        # 竞争因子：基于相对价值
        if target_value > avg_alternative_value:
            return 1.1  # 更有吸引力，可以定更高价格
        else:
            return 0.9  # 竞争力较弱，需要降低价格

    def optimize_prices_batch(self, items_data: List[Tuple[int, float, float]],
                              learning_rate: float = 0.01, epochs: int = 100) -> Dict[int, float]:
        """批量优化价格（梯度下降方法）"""
        prices = {item_id: (value + cost) / 2 for item_id, value, cost in items_data}

        for epoch in range(epochs):
            total_gradient = 0

            for item_id, value, cost in items_data:
                current_price = prices[item_id]

                # 计算收益函数梯度
                demand = self.demand_function(current_price, value)
                gradient = demand + (current_price - cost) * self.demand_derivative(current_price, value)

                # 价格更新
                new_price = current_price + learning_rate * gradient
                new_price = max(new_price, cost * 1.05)  # 保持最小利润

                prices[item_id] = new_price
                total_gradient += abs(gradient)

            # 检查收敛
            if total_gradient / len(items_data) < 0.001:
                print(f"价格优化在 {epoch + 1} 轮后收敛")
                break

        return prices

    def demand_derivative(self, price: float, value: float, k: float = None) -> float:
        """需求函数的导数"""
        if k is None:
            k = self.params.k
        demand = self.demand_function(price, value, k)
        return -k * demand * (1 - demand)


class ImprovedPOTProtocol:
    """改进的定价不经意传输协议"""

    def __init__(self, num_items: int = 10, price_range: Tuple[float, float] = (10, 100)):
        self.num_items = num_items
        self.price_range = price_range

        # 初始化各个组件
        self.ntru_params = NTRUParameters()
        self.privacy_params = PrivacyParameters()
        self.game_params = GameTheoryParameters()

        self.ntru = NTruCryptosystem(self.ntru_params)
        self.dp_engine = DifferentialPrivacyEngine(self.privacy_params)
        self.pricing_engine = GameTheoryPricing(self.game_params)

        # 参与者密钥
        self.buyer_keys = None  # (public_key, private_key_f, private_key_fp)
        self.seller_keys = None

        # 商品数据库
        self.item_database = []

        # 博弈均衡价格
        self.equilibrium_prices = {}

        # 交易记录
        self.transaction_history = []

        # 初始化统计
        self.stats = {
            'total_transactions': 0,
            'successful_transactions': 0,
            'failed_transactions': 0,
            'privacy_budget_used': 0.0,
            'total_processing_time': 0.0
        }

    def initialize_protocol(self):
        """初始化协议"""
        print("正在初始化改进POT协议...")

        # 1. 生成NTRU密钥对
        print("生成NTRU密钥对...")
        self.buyer_keys = self.ntru.generate_key_pair()
        self.seller_keys = self.ntru.generate_key_pair()

        # 2. 初始化商品数据库
        print("初始化商品数据库...")
        self._initialize_item_database()

        # 3. 计算博弈均衡价格
        print("计算博弈均衡价格...")
        self._calculate_equilibrium_prices()

        print("协议初始化完成！")
        self._print_initialization_summary()

    def _initialize_item_database(self):
        """初始化商品数据库"""
        # 生成模拟商品数据
        for i in range(1, self.num_items + 1):
            name = f"商品{i}"
            cost = np.random.uniform(5, 30)  # 成本
            value = cost * np.random.uniform(1.5, 3.0)  # 买家估值

            self.item_database.append({
                'item_id': i,
                'name': name,
                'cost': cost,
                'value': value,
                'description': f"这是商品{i}的描述"
            })

    def _calculate_equilibrium_prices(self):
        """计算博弈均衡价格"""
        items_data = []
        for item in self.item_database:
            items_data.append((item['item_id'], item['value'], item['cost']))

        self.equilibrium_prices = self.pricing_engine.calculate_nash_equilibrium(items_data)

    def _print_initialization_summary(self):
        """打印初始化摘要"""
        print("\n=== 协议初始化摘要 ===")
        print(f"商品数量: {self.num_items}")
        print(f"价格范围: {self.price_range}")
        print(f"NTRU参数: N={self.ntru_params.N}, p={self.ntru_params.p}, q={self.ntru_params.q}")
        print(f"隐私预算: ε={self.privacy_params.epsilon}")
        print(f"博弈参数: k={self.game_params.k}, α={self.game_params.alpha}")
        print("\n前5个商品的均衡价格:")
        for i in range(1, min(6, self.num_items + 1)):
            if i in self.equilibrium_prices:
                cost = next(item['cost'] for item in self.item_database if item['item_id'] == i)
                print(f"  商品{i}: 成本={cost:.2f}, 均衡价格={self.equilibrium_prices[i]:.2f}")

    def buyer_generate_request(self, item_id: int, quantity: int) -> Tuple[np.ndarray, Dict]:
        """买家生成加密请求"""
        start_time = time.time()

        # 验证请求参数
        if item_id < 1 or item_id > self.num_items:
            raise ValueError(f"无效的商品ID: {item_id}")
        if quantity <= 0:
            raise ValueError(f"无效的购买数量: {quantity}")

        # 构造请求消息
        request_data = {
            'item_id': item_id,
            'quantity': quantity,
            'timestamp': time.time(),
            'nonce': random.getrandbits(64)
        }

        # 编码为JSON字符串
        request_json = json.dumps(request_data)

        # 使用卖家的公钥加密请求
        seller_public_key = self.seller_keys[0]
        encrypted_request = self.ntru.encrypt(request_json, seller_public_key)

        processing_time = time.time() - start_time

        request_info = {
            'request_data': request_data,
            'processing_time': processing_time,
            'request_size': len(encrypted_request)
        }

        return encrypted_request, request_info

    def seller_process_request(self, encrypted_request: np.ndarray,
                               buyer_balance: float = 1000.0) -> Tuple[Dict, Dict]:
        """卖家处理买家请求"""
        start_time = time.time()

        try:
            # 1. 解密请求
            seller_private_key_f = self.seller_keys[1]
            seller_private_key_fp = self.seller_keys[2]

            decrypted_message = self.ntru.decrypt(encrypted_request,
                                                  seller_private_key_f,
                                                  seller_private_key_fp)

            request_data = json.loads(decrypted_message)
            item_id = request_data['item_id']
            quantity = request_data['quantity']

            # 2. 验证请求有效性
            validation_result = self._validate_request(item_id, quantity, buyer_balance)
            if not validation_result['is_valid']:
                return {
                    'status': 'rejected',
                    'reason': validation_result['reason']
                }, {'processing_time': time.time() - start_time}

            # 3. 获取商品信息
            item_info = next((item for item in self.item_database if item['item_id'] == item_id), None)
            if not item_info:
                return {
                    'status': 'rejected',
                    'reason': 'item_not_found'
                }, {'processing_time': time.time() - start_time}

            # 4. 应用差分隐私保护价格
            true_price = self.equilibrium_prices[item_id]
            noisy_price, privacy_log = self.dp_engine.laplace_mechanism(true_price)

            # 确保价格合理
            noisy_price = max(noisy_price, item_info['cost'] * 1.05)

            # 5. 计算交易金额
            total_amount = noisy_price * quantity

            # 6. 准备响应
            response = {
                'status': 'accepted',
                'item_id': item_id,
                'item_name': item_info['name'],
                'quantity': quantity,
                'original_price': true_price,
                'noisy_price': noisy_price,
                'total_amount': total_amount,
                'privacy_budget_used': privacy_log['epsilon_used'],
                'noise_added': privacy_log['noise']
            }

            processing_time = time.time() - start_time

            # 7. 记录交易
            self._record_transaction(request_data, response, processing_time)

            return response, {
                'processing_time': processing_time,
                'privacy_log': privacy_log
            }

        except Exception as e:
            return {
                'status': 'error',
                'reason': str(e)
            }, {'processing_time': time.time() - start_time}

    def _validate_request(self, item_id: int, quantity: int, buyer_balance: float) -> Dict:
        """验证请求有效性"""
        if item_id < 1 or item_id > self.num_items:
            return {'is_valid': False, 'reason': 'invalid_item_id'}

        if quantity <= 0:
            return {'is_valid': False, 'reason': 'invalid_quantity'}

        if item_id not in self.equilibrium_prices:
            return {'is_valid': False, 'reason': 'price_not_calculated'}

        # 检查余额是否充足
        estimated_cost = self.equilibrium_prices[item_id] * quantity
        if buyer_balance < estimated_cost:
            return {'is_valid': False, 'reason': 'insufficient_balance'}

        # 检查均衡价格是否合理
        item_info = next((item for item in self.item_database if item['item_id'] == item_id), None)
        if item_info and self.equilibrium_prices[item_id] < item_info['cost']:
            return {'is_valid': False, 'reason': 'price_below_cost'}

        return {'is_valid': True, 'reason': 'valid'}

    def _record_transaction(self, request_data: Dict, response: Dict, processing_time: float):
        """记录交易"""
        transaction_record = {
            'transaction_id': len(self.transaction_history) + 1,
            'timestamp': time.time(),
            'request': request_data,
            'response': response,
            'processing_time': processing_time
        }

        self.transaction_history.append(transaction_record)

        # 更新统计
        self.stats['total_transactions'] += 1
        if response['status'] == 'accepted':
            self.stats['successful_transactions'] += 1
        else:
            self.stats['failed_transactions'] += 1

        self.stats['privacy_budget_used'] += response.get('privacy_budget_used', 0)
        self.stats['total_processing_time'] += processing_time

    def simulate_transaction(self, item_id: int, quantity: int, buyer_balance: float = 1000.0):
        """模拟完整交易流程"""
        print(f"\n=== 模拟交易: 商品{item_id}, 数量{quantity} ===")

        try:
            # 买家生成请求
            print("1. 买家生成加密请求...")
            encrypted_request, request_info = self.buyer_generate_request(item_id, quantity)
            print(f"   请求生成时间: {request_info['processing_time'] * 1000:.2f}ms")

            # 卖家处理请求
            print("2. 卖家处理请求...")
            response, process_info = self.seller_process_request(encrypted_request, buyer_balance)

            if response['status'] == 'accepted':
                print(f"   交易成功!")
                print(f"   商品: {response['item_name']}")
                print(f"   数量: {response['quantity']}")
                print(f"   原始价格: {response['original_price']:.2f}")
                print(f"   含噪声价格: {response['noisy_price']:.2f}")
                print(f"   添加噪声: {response['noise_added']:.2f}")
                print(f"   总金额: {response['total_amount']:.2f}")
                print(f"   隐私预算使用: {response['privacy_budget_used']:.4f}")
                print(f"   处理时间: {process_info['processing_time'] * 1000:.2f}ms")
            else:
                print(f"   交易失败: {response['reason']}")

        except Exception as e:
            print(f"   交易异常: {str(e)}")

    def run_performance_test(self, num_transactions: int = 10):
        """运行性能测试"""
        print(f"\n=== 运行性能测试 ({num_transactions}次交易) ===")

        test_results = []
        total_time = 0

        for i in range(num_transactions):
            # 随机选择商品和数量
            item_id = random.randint(1, self.num_items)
            quantity = random.randint(1, 5)

            start_time = time.time()

            try:
                encrypted_request, _ = self.buyer_generate_request(item_id, quantity)
                response, process_info = self.seller_process_request(encrypted_request)

                test_time = time.time() - start_time
                total_time += test_time

                test_results.append({
                    'transaction_id': i + 1,
                    'item_id': item_id,
                    'quantity': quantity,
                    'status': response['status'],
                    'processing_time': test_time,
                    'privacy_used': response.get('privacy_budget_used', 0) if response['status'] == 'accepted' else 0
                })

            except Exception as e:
                print(f"交易{i + 1}失败: {str(e)}")

        # 打印测试结果
        self._print_performance_summary(test_results, total_time)

        return test_results

    def _print_performance_summary(self, test_results: List[Dict], total_time: float):
        """打印性能测试摘要"""
        successful = [r for r in test_results if r['status'] == 'accepted']
        failed = [r for r in test_results if r['status'] != 'accepted']

        print("\n=== 性能测试摘要 ===")
        print(f"总交易数: {len(test_results)}")
        print(f"成功交易: {len(successful)}")
        print(f"失败交易: {len(failed)}")
        print(f"总时间: {total_time:.3f}秒")
        print(f"平均交易时间: {total_time / len(test_results) * 1000:.2f}ms")

        if successful:
            avg_privacy_used = sum(r['privacy_used'] for r in successful) / len(successful)
            print(f"平均隐私预算使用: {avg_privacy_used:.4f}")
            print(f"总隐私预算使用: {sum(r['privacy_used'] for r in successful):.4f}")

        # 打印时间分布
        processing_times = [r['processing_time'] * 1000 for r in test_results]
        print(f"最短时间: {min(processing_times):.2f}ms")
        print(f"最长时间: {max(processing_times):.2f}ms")
        print(f"时间标准差: {np.std(processing_times):.2f}ms")

    def run_privacy_analysis(self, num_trials: int = 100):
        """运行隐私保护分析"""
        print(f"\n=== 隐私保护分析 ({num_trials}次试验) ===")

        true_prices = []
        noisy_prices = []
        noises = []

        # 测试不同隐私预算下的效果
        epsilon_values = [0.01, 0.05, 0.1, 0.5, 1.0]

        for epsilon in epsilon_values:
            # 临时设置隐私预算
            self.dp_engine.params.epsilon = epsilon
            self.dp_engine.epsilon_used = 0

            trial_results = []

            for _ in range(num_trials):
                # 随机选择一个均衡价格
                if self.equilibrium_prices:
                    random_item_id = random.choice(list(self.equilibrium_prices.keys()))
                    true_price = self.equilibrium_prices[random_item_id]

                    # 应用差分隐私
                    noisy_price, privacy_log = self.dp_engine.laplace_mechanism(true_price)

                    trial_results.append({
                        'true_price': true_price,
                        'noisy_price': noisy_price,
                        'noise': privacy_log['noise'],
                        'relative_error': abs(noisy_price - true_price) / true_price
                    })

            if trial_results:
                avg_relative_error = np.mean([r['relative_error'] for r in trial_results])
                max_relative_error = np.max([r['relative_error'] for r in trial_results])

                print(f"\nε={epsilon}:")
                print(f"  平均相对误差: {avg_relative_error * 100:.1f}%")
                print(f"  最大相对误差: {max_relative_error * 100:.1f}%")

    def visualize_results(self):
        """可视化结果"""
        if not self.transaction_history:
            print("没有足够的交易数据进行可视化")
            return

        # 准备数据
        transaction_ids = [t['transaction_id'] for t in self.transaction_history]
        processing_times = [t['processing_time'] * 1000 for t in self.transaction_history]

        accepted_transactions = [t for t in self.transaction_history if t['response']['status'] == 'accepted']

        if accepted_transactions:
            original_prices = [t['response']['original_price'] for t in accepted_transactions]
            noisy_prices = [t['response']['noisy_price'] for t in accepted_transactions]
            privacy_used = [t['response']['privacy_budget_used'] for t in accepted_transactions]

            # 创建图形
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))

            # 1. 处理时间分布
            axes[0, 0].plot(transaction_ids, processing_times, 'b-o', markersize=4)
            axes[0, 0].set_xlabel('交易ID')
            axes[0, 0].set_ylabel('处理时间 (ms)')
            axes[0, 0].set_title('交易处理时间分布')
            axes[0, 0].grid(True, alpha=0.3)

            # 2. 价格比较
            indices = list(range(len(original_prices)))
            width = 0.35
            axes[0, 1].bar([i - width / 2 for i in indices], original_prices, width, label='原始价格', alpha=0.7)
            axes[0, 1].bar([i + width / 2 for i in indices], noisy_prices, width, label='含噪声价格', alpha=0.7)
            axes[0, 1].set_xlabel('交易序号')
            axes[0, 1].set_ylabel('价格')
            axes[0, 1].set_title('差分隐私价格保护效果')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)

            # 3. 隐私预算使用
            axes[1, 0].plot(range(len(privacy_used)), np.cumsum(privacy_used), 'g-s', markersize=4)
            axes[1, 0].set_xlabel('交易次数')
            axes[1, 0].set_ylabel('累计隐私预算使用')
            axes[1, 0].set_title('隐私预算消耗情况')
            axes[1, 0].grid(True, alpha=0.3)

            # 4. 均衡价格分布
            if self.equilibrium_prices:
                item_ids = list(self.equilibrium_prices.keys())[:10]  # 只显示前10个
                eq_prices = [self.equilibrium_prices[i] for i in item_ids]
                item_costs = [next(item['cost'] for item in self.item_database if item['item_id'] == i) for i in
                              item_ids]

                x = np.arange(len(item_ids))
                width = 0.35
                axes[1, 1].bar(x - width / 2, item_costs, width, label='成本', alpha=0.7)
                axes[1, 1].bar(x + width / 2, eq_prices, width, label='均衡价格', alpha=0.7)
                axes[1, 1].set_xlabel('商品ID')
                axes[1, 1].set_ylabel('价格')
                axes[1, 1].set_title('博弈均衡价格分布')
                axes[1, 1].set_xticks(x)
                axes[1, 1].set_xticklabels(item_ids)
                axes[1, 1].legend()
                axes[1, 1].grid(True, alpha=0.3)

            plt.tight_layout()
            plt.show()

    def get_protocol_statistics(self) -> Dict:
        """获取协议统计信息"""
        return {
            'total_transactions': self.stats['total_transactions'],
            'successful_transactions': self.stats['successful_transactions'],
            'failed_transactions': self.stats['failed_transactions'],
            'success_rate': self.stats['successful_transactions'] / max(1, self.stats['total_transactions']),
            'avg_processing_time': self.stats['total_processing_time'] / max(1, self.stats['total_transactions']),
            'total_privacy_budget_used': self.stats['privacy_budget_used'],
            'remaining_privacy_budget': self.privacy_params.epsilon - self.stats['privacy_budget_used']
        }


def main():
    """主函数：演示协议使用"""
    print("=" * 60)
    print("定价不经意传输协议(POT)改进方案复现代码")
    print("作者：齐轲 (20233001410)")
    print("=" * 60)

    # 创建协议实例
    protocol = ImprovedPOTProtocol(num_items=10, price_range=(10, 100))

    # 1. 初始化协议
    protocol.initialize_protocol()

    # 2. 模拟几个交易
    print("\n" + "=" * 60)
    print("模拟交易演示")
    print("=" * 60)

    # 模拟成功交易
    protocol.simulate_transaction(item_id=3, quantity=2)
    protocol.simulate_transaction(item_id=5, quantity=1)

    # 模拟失败交易（数量为0）
    try:
        protocol.simulate_transaction(item_id=2, quantity=0)
    except ValueError as e:
        print(f"\n预期失败交易: {str(e)}")

    # 3. 运行性能测试
    print("\n" + "=" * 60)
    print("性能测试")
    print("=" * 60)

    test_results = protocol.run_performance_test(num_transactions=20)

    # 4. 隐私保护分析
    print("\n" + "=" * 60)
    print("隐私保护分析")
    print("=" * 60)

    protocol.run_privacy_analysis(num_trials=50)

    # 5. 获取统计信息
    print("\n" + "=" * 60)
    print("协议统计信息")
    print("=" * 60)

    stats = protocol.get_protocol_statistics()
    for key, value in stats.items():
        if isinstance(value, float):
            if 'time' in key:
                print(f"{key}: {value * 1000:.2f}ms")
            elif 'rate' in key:
                print(f"{key}: {value * 100:.1f}%")
            else:
                print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")

    # 6. 可视化结果（可选）
    print("\n" + "=" * 60)
    print("生成可视化图表")
    print("=" * 60)

    try:
        protocol.visualize_results()
    except Exception as e:
        print(f"可视化生成失败: {e}")
        print("请确保已安装matplotlib库: pip install matplotlib")

    print("\n" + "=" * 60)
    print("演示完成！")
    print("=" * 60)


if __name__ == "__main__":
    # 运行演示
    main()

    # 单独运行各个组件的示例
    print("\n\n组件单独使用示例:")
    print("-" * 40)

    # 1. NTRU加密示例
    print("1. NTRU加密示例:")
    ntru_params = NTRUParameters(N=251, p=3, q=128, d=72)
    ntru = NTruCryptosystem(ntru_params)

    h, f, fp = ntru.generate_key_pair()
    message = "Hello, NTRU!"
    ciphertext = ntru.encrypt(message, h)
    decrypted = ntru.decrypt(ciphertext, f, fp)

    print(f"   原始消息: {message}")
    print(f"   解密消息: {decrypted}")
    print(f"   加解密成功: {message in decrypted}")

    # 2. 差分隐私示例
    print("\n2. 差分隐私示例:")
    privacy_params = PrivacyParameters(epsilon=0.1, sensitivity=1.0)
    dp_engine = DifferentialPrivacyEngine(privacy_params)

    true_value = 50.0
    noisy_value, privacy_log = dp_engine.laplace_mechanism(true_value)

    print(f"   真实值: {true_value}")
    print(f"   含噪声值: {noisy_value:.2f}")
    print(f"   添加噪声: {privacy_log['noise']:.2f}")
    print(f"   隐私预算使用: {privacy_log['epsilon_used']}")

    # 3. 博弈论定价示例
    print("\n3. 博弈论定价示例:")
    game_params = GameTheoryParameters(k=0.1, alpha=0.05)
    pricing_engine = GameTheoryPricing(game_params)

    # 计算单个商品的均衡价格
    item_id = 1
    value = 60.0
    cost = 20.0
    eq_price, iterations = pricing_engine.calculate_equilibrium_price(item_id, value, cost)

    print(f"   商品估值: {value}")
    print(f"   商品成本: {cost}")
    print(f"   均衡价格: {eq_price:.2f}")
    print(f"   迭代次数: {iterations}")

    print("\n" + "=" * 40)
    print("所有组件测试完成！")

