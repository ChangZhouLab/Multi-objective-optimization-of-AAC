import random
import joblib
import numpy as np
import pandas as pd
from deap import base, creator, tools


# 加载机器学习模型
model1 = joblib.load("regress_fc.pkl")
model2 = joblib.load("regress_po.pkl")
model3 = joblib.load("regress_co2.pkl")


# 定义目标函数
def evaluate(x):

    y1 = abs(model1.predict(np.array(x).reshape(1, -1))[0] - 40)
    y2 = model2.predict(np.array(x).reshape(1, -1))[0]
    y3 = model3.predict(np.array(x).reshape(1, -1))[0]

    return (-y1, -y2, -y3)


# 创建一个多目标优化问题
creator.create("FitnessMulti", base.Fitness, weights=(100.0, 1.0, 1.0))
creator.create("Individual", list, fitness=creator.FitnessMulti)

# 初始化工具箱
toolbox = base.Toolbox()
# 注册基因，个体和种群
toolbox.register("attr_float", random.uniform, 0, 1)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=13)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
# 注册评估函数，注册交叉&变异操作，选择操作
toolbox.register("evaluate", evaluate)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
toolbox.register("select", tools.selNSGA2)


all_individuals = []  # 存储所有个体的列表
best_individual = []  # 存储每一代最优个体的列表
best_objectives = []  # 存储每一代最优个体表现的列表
best_compressive = []  # 存储每一代最优个体抗压强度的列表

# 创建样本数为100的种群, 迭代50次
population = toolbox.population(n=100)

for generation in range(50):

    # 评估种群中的个体
    fitnesses = toolbox.map(toolbox.evaluate, population)
    for ind, fit in zip(population, fitnesses):
        ind.fitness.values = fit

    # 通过克隆生成下一代种群，以便后续用于交叉、变异
    offspring = toolbox.select(population, len(population))
    offspring = [toolbox.clone(ind) for ind in offspring]
    all_individuals.append(population[:])  # 记录每一代的种群

    # 基因交叉互换和突变
    for child1, child2 in zip(offspring[::2], offspring[1::2]):
        if random.random() < 0.5:
            toolbox.mate(child1, child2)

        # child1[1] = 0.1
        # child1[1] = 2 * child2[0]

        child1[0] = max(min(child1[0], 0.73), 0.22)
        child1[1] = max(min(child1[1], 0.60), 0.00)
        child1[2] = max(min(child1[2], 0.24), 0.04)
        child1[3] = max(min(child1[3], 0.43), 0.15)
        child1[4] = max(min(child1[4], 0.24), 0.12)
        child1[5] = max(min(child1[5], 1.00), 0.13)
        child1[6] = max(min(child1[6], 0.91), 0.25)
        child1[7] = max(min(child1[7], 0.93), 0.08)
        child1[8] = max(min(child1[8], 0.65), 0.45)
        child1[9] = max(min(child1[9], 1.00), 0.26)
        child1[10] = max(min(child1[10], 0.05), 0.01)
        child1[11] = max(min(child1[11], 0.73), 0.20)
        child1[12] = max(min(child1[12], 0.75), 0.13)

        child2[0] = max(min(child2[0], 0.73), 0.22)
        child2[1] = max(min(child2[1], 0.60), 0.00)
        child2[2] = max(min(child2[2], 0.24), 0.04)
        child2[3] = max(min(child2[3], 0.43), 0.15)
        child2[4] = max(min(child2[4], 0.24), 0.12)
        child2[5] = max(min(child2[5], 1.00), 0.13)
        child2[6] = max(min(child2[6], 0.91), 0.25)
        child2[7] = max(min(child2[7], 0.93), 0.08)
        child2[8] = max(min(child2[8], 0.65), 0.45)
        child2[9] = max(min(child2[9], 1.00), 0.26)
        child2[10] = max(min(child2[10], 0.05), 0.01)
        child2[11] = max(min(child2[11], 0.73), 0.20)
        child2[12] = max(min(child2[12], 0.75), 0.13)

        # 变异会导致基因跳出限制范围
        # toolbox.mutate(child1)
        # toolbox.mutate(child2)

        del child1.fitness.values, child2.fitness.values

    # 将子代合并到父代中
    population[:] = offspring

    # 输出最优子代信息
    best_offspring = max(offspring, key=lambda ind: ind.fitness.values)
    best_fitnesses = toolbox.evaluate(best_offspring)
    fc = model1.predict(np.array(best_offspring).reshape(1, -1))[0]
    best_individual.append(best_offspring)
    best_objectives.append(best_fitnesses)
    best_compressive.append(fc)
    best_individual_save = pd.DataFrame(best_individual)
    best_objectives_save = pd.DataFrame(best_objectives)
    best_compressive_save = pd.DataFrame(best_compressive)
    best_individual_save.to_excel('best_individual.xlsx', index=False)
    best_objectives_save.to_excel('best_objectives.xlsx', index=False)
    best_compressive_save.to_excel('best_compressive.xlsx', index=False)



# 输出第一代,最后一代全部个体
first_generation = all_individuals[0]
last_generation = all_individuals[-1]
first_gen_individual = []
first_gen_objectives = []
last_gen_individual = []
last_gen_objectives = []


for ind in first_generation:
    objectives = evaluate(ind)
    first_gen_individual.append(ind)
    first_gen_objectives.append(objectives)
    first_gen_individual_save = pd.DataFrame(first_gen_individual)
    first_gen_objectives_save = pd.DataFrame(first_gen_objectives)
    first_gen_individual_save.to_excel('first_gen_individual.xlsx', index=False)
    first_gen_objectives_save.to_excel('first_gen_objectives.xlsx', index=False)

for ind in last_generation:
    objectives = evaluate(ind)
    last_gen_individual.append(ind)
    last_gen_objectives.append(objectives)
    last_gen_individual_save = pd.DataFrame(last_gen_individual)
    last_gen_objectives_save = pd.DataFrame(last_gen_objectives)
    last_gen_individual_save.to_excel('last_gen_individual.xlsx', index=False)
    last_gen_objectives_save.to_excel('last_gen_objectives.xlsx', index=False)



# 输出最优解
best_individuals = tools.selBest(population, 1)
best_solution = best_individuals[0]
print("Best Solution:", best_solution)

# 计算最优结果
best_fitnesses = toolbox.evaluate(best_solution)
print("Best Fitness:", best_fitnesses)

y1 = model1.predict(np.array(best_solution).reshape(1, -1))[0]
y2 = model2.predict(np.array(best_solution).reshape(1, -1))[0]
y3 = model3.predict(np.array(best_solution).reshape(1, -1))[0]
print(y1, y2, y3)




