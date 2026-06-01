import rospy
from visualization_msgs.msg import Marker, MarkerArray
from idmp_ros.srv import GetDistanceGradient
import numpy as np
import random

shotThreshold = 0.6
oldDist = [0,0,0,0,0,0,0,0,0,0,0]
originx = 0
originy = 0
originz = 0.5
textx = 4.0
texty = 0.0
textz = 2.0
ballPose = np.array([originx,originy,originz])
ballVel = 0
ballVelMax = 5
ballVelDamping = 0.95
ballDirection = np.array([0,0,0])
goalCounter = 0

updateRate = 0.05

goalPolePos = 1.0
fieldSize = [4,2.0,2]

#--------------------------------------
#caliboard AruCo setup

# |-------left--------|
# |                   |
# |                   |
# |down  base_link  up|
# |                   |
# |                   |
# |-------right-------|
#
#           ^  
#           |
#           |
#      camera_link
# (where the camera is)

#--------------------------------------
# play field setup

# |--------(-x)-------|
# |                   |
# |                   |
# |+y   base_link  -y |
# |                   |
# |                   |
# |--------(+x)-------|
#
#           ^  
#           |
#           |
#      camera_link
# (where the camera is)

# -------|goal!|-------

def createPole(x,y,z,y_scale,z_scale,idnum):
    m = Marker()
    m.header.frame_id = 'base_link'
    m.header.stamp = rospy.Time.now()
    m.id = idnum
    m.type = Marker.CUBE
    m.action = Marker.ADD
    m.pose.position.x = x
    m.pose.position.y = y
    m.pose.position.z = z
    m.pose.orientation.x = 0
    m.pose.orientation.y = 0
    m.pose.orientation.z = 0
    m.pose.orientation.w = 1
    m.scale.x = 0.05
    m.scale.y = y_scale
    m.scale.z = z_scale
    m.color.r = 0
    m.color.g = 0
    m.color.b = 0
    m.color.a = 1
    return m

def createBall(x,y,z,idnum):
    m = Marker()
    m.header.frame_id = 'base_link'
    m.header.stamp = rospy.Time.now()
    m.id = idnum
    m.type = Marker.SPHERE
    m.action = Marker.ADD
    m.pose.position.x = x
    m.pose.position.y = y
    m.pose.position.z = z
    m.pose.orientation.x = 0
    m.pose.orientation.y = 0
    m.pose.orientation.z = 0
    m.pose.orientation.w = 1
    m.scale.x = 0.5
    m.scale.y = 0.5
    m.scale.z = 0.5
    m.color.r = 1
    m.color.g = 0
    m.color.b = 0
    m.color.a = 1
    return m

def createGoalText(x, y, z, idnum, msg):
    m = Marker()
    m.header.frame_id = 'base_link'
    m.header.stamp = rospy.Time.now()
    m.ns = "goal_text"
    m.id = idnum
    m.type = Marker.TEXT_VIEW_FACING
    m.action = Marker.ADD
    m.pose.position.x = x
    m.pose.position.y = y
    m.pose.position.z = z + 1.0
    m.pose.orientation.w = 1.0
    m.scale.z = 1.0
    m.color.r = 1.0
    m.color.g = 1.0
    m.color.b = 0.0
    m.color.a = 1.0
    m.text = msg
    m.lifetime = rospy.Duration(3.0)  # visible for 3 seconds
    return m

def createGoalEffect(x, y, z, idnum, scale=1.0):
    m = Marker()
    m.header.frame_id = 'base_link'
    m.header.stamp = rospy.Time.now()
    m.ns = "goal_effect"
    m.id = idnum
    m.type = Marker.SPHERE
    m.action = Marker.ADD
    m.pose.position.x = x
    m.pose.position.y = y
    m.pose.position.z = z
    m.pose.orientation.w = 1.0
    m.scale.x = scale
    m.scale.y = scale
    m.scale.z = scale
    m.color.r = 1.0
    m.color.g = 1.0
    m.color.b = 0.0
    m.color.a = 0.6  # translucent
    m.lifetime = rospy.Duration(3.0)  # auto-delete after 1 sec
    return m

def createGoalFireworks(x, y, z, count=20):
    arr = MarkerArray()
    for i in range(count):
        m = Marker()
        m.header.frame_id = 'base_link'
        m.header.stamp = rospy.Time.now()
        m.ns = "goal_fireworks"
        m.id = i
        m.type = Marker.SPHERE
        m.action = Marker.ADD

        # random offset from center
        offset = np.random.uniform(-1.0, 1.0, 3)
        offset[2] = abs(offset[2]) + 0.2  # ensure it's upward

        m.pose.position.x = x + offset[0]
        m.pose.position.y = y + offset[1]
        m.pose.position.z = z + offset[2]

        m.scale.x = m.scale.y = m.scale.z = random.uniform(0.1, 0.3)

        # random color
        m.color.r = random.random()
        m.color.g = random.random()
        m.color.b = random.random()
        m.color.a = 0.9

        m.lifetime = rospy.Duration(3.0)
        arr.markers.append(m)
    return arr

def setupGoal():
    print("SetupGoal")
    goal_pub = rospy.Publisher("goal", MarkerArray, queue_size=0, latch=True)
    mArr = MarkerArray()
    mArr.markers.append(createPole(fieldSize[0],goalPolePos,fieldSize[2]/2,0.05,fieldSize[2],1))
    mArr.markers.append(createPole(fieldSize[0],-goalPolePos,fieldSize[2]/2,0.05,fieldSize[2],2))
    mArr.markers.append(createPole(fieldSize[0],0,fieldSize[2],goalPolePos*2,0.05,3))
    goal_pub.publish(mArr)

def calcBallParams(dist,grad):
    global ballVel, ballDirection, oldDist
    delta =0
    if ballVel < 0.05:
        ballVel = 0
    if ballVel > 0:
        # ToDO: Coole Funktion
        ballVel = ballVel*ballVelDamping

    # if(dist < 0.4):
    #     grad[2] = 0
    #     ballVel = 1/dist
    # else:
    #     grad = np.array([0,0,0])
    # ballDirection = grad / max(dist, 1e-5)
        
    # delta =0
    # if dist < shotThreshold:
    #     delta = oldDist[0]-dist
    #     if delta > 0.2:
    #         ballVel = min(ballVel+200*(delta),ballVelMax)
    #         ballDirection = grad
    #         ballDirection[2] = 0
    # if(dist < 100):
    #     oldDist.append(dist)
    #     oldDist.pop(0)

    # --- distance change since last cycle ---
    delta = 0.0
    if oldDist:                     # buffer not empty
        delta = oldDist[-1] - dist  # previous – current
    if(dist < 100):
        oldDist.append(dist)        # keep newest value
        oldDist.pop(0)
    if len(oldDist) > 11:
        oldDist.pop(0)

    # --- “kick” the ball ---
    if dist < shotThreshold and delta > 0.1:   # smaller delta works better
        ballVel = min(ballVel + 200 * delta, ballVelMax)
        #if np.linalg.norm(grad) > 1e-6:
        ballDirection = grad/np.linalg.norm(grad)  # normalise
        ballDirection[2] = 0.0

    #print("Vell:", ballVel, "delta:",delta, oldDist)
    print("Vell:", ballVel, "delta:", delta, "dist", dist)
    return


def calcNewBallPose():
    global ballPose
    ballPose = ballPose+(updateRate*ballVel)*ballDirection
    # ballPose = ballPose+updateRate*ballDirection
    return
    
def updateVis():
    ball_pub = rospy.Publisher("Ball", Marker, queue_size=0)
    ball_pub.publish(createBall(ballPose[0],ballPose[1],originz,99))
    return

def goalQuery():
    global ballPose,ballVel,ballDirection
    #goal
    if ballPose[0] > (fieldSize[0]-0.2) and ballPose[1] < goalPolePos and ballPose[1] > -goalPolePos:
        #goalCounter+=1
        print("GOAL !!!!!")
        #print(goalCounter)
        goal_effect_pub.publish(createGoalEffect(ballPose[0], ballPose[1], ballPose[2], idnum=777, scale=1.5))
        goal_text_pub.publish(createGoalText(textx, texty, textz, idnum=778, msg="GOAL !!!"))
        goal_fireworks_pub.publish(createGoalFireworks(ballPose[0], ballPose[1], ballPose[2]))
    
        rospy.sleep(5.0)  # <--- Simple 5 second pause
        ballPose = np.array([originx,originy,originz])
        ballVel = 0
        ballDirection = np.array([0,0,0])

    #out
    if ballPose[0] > fieldSize[0] or ballPose[0] < -0.5 or ballPose[1] > fieldSize[1] or ballPose[1] < -fieldSize[1]:
        print("OUT !!!!!")
        goal_text_pub.publish(createGoalText(textx, texty, textz, idnum=778, msg="OUT (*_*) !!!"))

        rospy.sleep(5.0)  # <--- Simple 5 second pause
        ballPose = np.array([originx,originy,originz])
        ballVel = 0
        ballDirection = np.array([0,0,0])

    if(ballVel==0):
        #print("DO IT !!!!!")
        goal_text_pub.publish(createGoalText(textx, texty, textz, idnum=778, msg="PUSH IT !!!"))
        ballPose = np.array([originx,originy,originz])
        ballVel = 0
        ballDirection = np.array([0,0,0])
    return

if __name__=="__main__":
    rospy.init_node("soccer")
    query = rospy.ServiceProxy('query_dist_field', GetDistanceGradient)
    goal_effect_pub = rospy.Publisher("goal_effect", Marker, queue_size=10)
    goal_text_pub = rospy.Publisher("goal_text", Marker, queue_size=10, latch=True)
    goal_fireworks_pub = rospy.Publisher("goal_fireworks", MarkerArray, queue_size=10)

    #goal_pub = rospy.Publisher("goal", Marker, queue_size=0)
    setupGoal()
    while not rospy.is_shutdown():
        response = query(ballPose)
        dist = response.distances[0]
        grad = np.array(response.gradients)
        # print("Dist: ", dist," Grad: ",grad)
        calcBallParams(dist,grad)

        calcNewBallPose()
        updateVis()
        goalQuery()
        rospy.sleep(updateRate)
