from argParser import *
import os
import subprocess
import psutil
from env import *

#given current state and goal, generate problem.pddl
def generateProblemFile(loc_pro, iskey, quality):
    problemfile = open(args.problem_file,"w")
    problemfile.truncate()
    init_prob = open("initial_prob.pddl","r")
    for line in init_prob:
        problemfile.write(line)
    init_prob.close()
    if(iskey):
        problemfile.write("\t(keyexist)\n")       
    problemfile.write("\t(at %s)\n" % loc_pro)
    problemfile.write("\t(= (quality) %d) )\n" % quality)
    problemfile.write(")\n")



def generatePlan1(domainfile=args.template_domain_file,problemfile = args.template_problem_file):
    plan = []
    ming = "timeout 5s ./Metric/ff -o "+domainfile+" -f "+problemfile
    #./Metric/ff -o Metric/Test/templateDomain.pddl -f Metric/Test/templateProblem.pddl
    p = os.popen(ming)
    print(p)
    flag = False
    cost = 0
    for i in p.readlines():
        if i[:11]=="plan cost: ":
            cost=i[11:]
            flag = False
        if flag:
            a = i[11:]
            plan.append(a.split())
        if i=="ff: found legal plan as follows\n":
            flag = True    
    return plan, cost


def generatePlan(domainfile=args.template_domain_file,problemfile = args.template_problem_file):
    plan = []
    ming = "timeout 5s ./Metric/ff -o "+domainfile+" -f "+problemfile + " > result.txt"
    #./Metric/ff -o Metric/Test/templateDomain.pddl -f Metric/Test/templateProblem.pddl
    plannerProcess = subprocess.Popen(ming, shell=True)
    p = psutil.Process(plannerProcess.pid)
    try:
        p.wait(timeout = 180)
    except psutil.TimeoutExpired:
        p.kill()
        print("Planning timeout. Process killed.")
        return [], 0
    
    flag = False
    cost = 0
    p = open('result.txt')
    for i in p.readlines():
        if i[:11]=="plan cost: ":
            cost=i[11:]
            flag = False
        if flag:
            a = i[11:]
            plan.append(a.split())
        if i=="ff: found legal plan as follows\n":
            flag = True    
    return plan, cost


def plan2Lower(plan):
    for i in range(len(plan) ):
        for j in range(len(plan[i]) ):
            plan[i][j] = plan[i][j].lower()
    return plan

def testplanner():
    plan,cost = generatePlan(args.template_domain_file,args.template_problem_file)
    print(plan)
    print("cost is : ",cost)
    
    print("lower plan")
    print(plan2Lower(plan))
        
testplanner()