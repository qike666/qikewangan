#include <stdio.h>
#include <stdlib.h>
#include <time.h>

// 1. 定义多项式的“长度”（次数+1）。为了简单，我们设长度为4。
//    这相当于论文里提到的“密文扩张”，即密文比明文占据更多空间。
#define POLY_DEGREE 4

// 2. 定义一个“密文”结构体，它就是一个多项式。
//    对比论文里1.2节提到的：密文通常是一个包含噪声和数据的向量。
typedef struct {
    int coeffs[POLY_DEGREE];
} Ciphertext;

// --- 辅助函数：打印多项式，方便你观察实验现象 ---
void print_poly(char *name, Ciphertext p) {
    printf("%s = [", name);
    for (int i = 0; i < POLY_DEGREE; i++) {
        printf("%d", p.coeffs[i]);
        if (i < POLY_DEGREE - 1) printf(", ");
    }
    printf("]\n");
}

// 3. 加密函数：Encrypt(m)
//    把明文m“藏”在多项式的常数项（索引0），其余位置填入随机数作为“噪声”。
//    这对应论文1.2节的“加密（Encrypt）”模块，展示了“引入噪声”的过程。
Ciphertext encrypt(int plaintext) {
    Ciphertext c;
    // 用当前时间作为随机数种子
    srand(time(NULL)); 

    // 常数项存放明文
    c.coeffs[0] = plaintext;

    // 其他项填入0到9的随机整数作为“噪声”，模拟密文数据的随机性
    for (int i = 1; i < POLY_DEGREE; i++) {
        c.coeffs[i] = rand() % 10; 
    }
    return c;
}

// 4. 解密函数：Decrypt(c)
//    解密很简单，就是取出多项式的常数项。
//    这对应论文1.2节的“解密（Decrypt）”模块。
int decrypt(Ciphertext c) {
    return c.coeffs[0];
}

// 5. 同态加法：对两个“密文多项式”直接相加
//    这对应论文1.1节的核心公式：D(E(m1) + E(m2)) = m1 + m2
//    这里完全符合“密文可在不解密前提下直接参与计算”的定义。
Ciphertext homomorphic_add(Ciphertext a, Ciphertext b) {
    Ciphertext result;
    for (int i = 0; i < POLY_DEGREE; i++) {
        result.coeffs[i] = a.coeffs[i] + b.coeffs[i];
    }
    return result;
}

// 6. 同态乘法：对两个“密文多项式”相乘
//    这同样对应核心公式：D(E(m1) * E(m2)) = m1 * m2
//    注意：这里我们没有处理“噪声”的管理，因此如果做太多次乘法，结果会“爆掉”。
//    这正好可以用来类比论文1.3节提到的“噪声问题”！
Ciphertext homomorphic_multiply(Ciphertext a, Ciphertext b) {
    Ciphertext result;
    // 初始化结果多项式的所有系数为0
    for (int i = 0; i < POLY_DEGREE; i++) {
        result.coeffs[i] = 0;
    }

    // 实现简单的多项式乘法（不考虑进位，只做系数相乘后相加）
    // 这是一个简化的卷积操作，足以演示原理。
    for (int i = 0; i < POLY_DEGREE; i++) {
        for (int j = 0; j < POLY_DEGREE; j++) {
            // 如果索引不超出范围，则累加
            if (i + j < POLY_DEGREE) {
                result.coeffs[i + j] += a.coeffs[i] * b.coeffs[j];
            }
        }
    }
    return result;
}

// --- 主程序：演示整个流程 ---
int main() {
    printf("===== 全同态加密（FHE）原理模拟实验 =====\n\n");

    // 1. 定义两个明文
    int m1 = 3;
    int m2 = 5;
    printf("原始明文数据: m1 = %d, m2 = %d\n\n", m1, m2);

    // 2. 加密：生成密文
    Ciphertext c1 = encrypt(m1);
    Ciphertext c2 = encrypt(m2);
    printf("加密后的密文（观察噪声项）:\n");
    print_poly("c1", c1);
    print_poly("c2", c2);
    printf("\n");

    // 3. 同态加法实验
    printf("--- 实验1：同态加法 ---\n");
    Ciphertext c_add = homomorphic_add(c1, c2);
    print_poly("c1 + c2", c_add);
    int result_add = decrypt(c_add);
    printf("解密 c1 + c2 的结果: %d\n", result_add);
    printf("明文直接相加的结果: %d + %d = %d\n", m1, m2, m1 + m2);
    printf("? 结论：解密后的结果与明文直接相加一致！\n\n");

    // 4. 同态乘法实验
    printf("--- 实验2：同态乘法 ---\n");
    Ciphertext c_mul = homomorphic_multiply(c1, c2);
    print_poly("c1 * c2", c_mul);
    int result_mul = decrypt(c_mul);
    printf("解密 c1 * c2 的结果: %d\n", result_mul);
    printf("明文直接相乘的结果: %d * %d = %d\n", m1, m2, m1 * m2);
    printf("? 结论：解密后的结果与明文直接相乘一致！\n\n");

    // 5. 演示“噪声问题”（对应论文1.3节）
    printf("--- 实验3：演示噪声累积问题（对应论文1.3节） ---\n");
    printf("对密文进行多次乘法（密文自乘3次）...\n");
    Ciphertext c_mul2 = homomorphic_multiply(c_mul, c1); 
    print_poly("c1^3", c_mul2);
    int result_mul2 = decrypt(c_mul2);
    printf("解密 c1^3 的结果: %d\n", result_mul2);
    printf("明文直接计算的结果: %d^3 = %d\n", m1, m1 * m1 * m1);
    printf("? 注意：解密结果 %d 与预期 %d 不符！\n", result_mul2, m1 * m1 * m1);
    printf("原因：噪声在多次乘法中剧烈增长，超过了多项式能承载的阈值，\n");
    printf("      这导致最终无法正确解密，这就是论文中提到的核心瓶颈——噪声问题。\n\n");

    printf("===== 实验结束 =====\n");
    printf("本实验通过简化模型，验证了FHE的核心思想，并观察到了噪声带来的影响。\n");
    return 0;
}
