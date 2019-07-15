#include "ros/ros.h"
#include "std_msgs/String.h"
#include "geometry_msgs/Twist.h"

#define FORWARD "Forward"
#define RIGHT "TurnRight"
#define LEFT "TurnLeft"
#define NONE "None"

void callback(const boost::shared_ptr<std_msgs::String const> msg, double& VITESSE_ROBOT, double& alpha, double& target_linear_speed, double& target_angular_speed) {
    std::string command = msg->data;

    if (command == FORWARD) {
        target_linear_speed = VITESSE_ROBOT;
        target_angular_speed = 0;
    } else if (command == RIGHT) {
        target_angular_speed = -3*VITESSE_ROBOT;
    } else if (command == LEFT) {
        target_angular_speed = 3*VITESSE_ROBOT;
    } else if (command == NONE) {
        target_angular_speed = 0;
        target_linear_speed = 0;
    } else {
        target_angular_speed = 0;
        target_linear_speed = 0;
        ROS_WARN("Unknown command [%s]", command.c_str());
    }
}

int main(int argc, char **argv) {
    double VITESSE_ROBOT = 0.3;
    double alpha = 0.1;
    ros::init(argc, argv, "khepera_control");
    ros::NodeHandle n;
    ros::Rate loop_rate(4);

    double current_linear_speed = 0;
    double current_angular_speed = 0;
    double target_linear_speed = 0;
    double target_angular_speed = 0;

    ros::Publisher pub = n.advertise<geometry_msgs::Twist>("/cmd_vel", 1000);
    ros::Subscriber sub = n.subscribe<std_msgs::String>("/khepera/string_commands", 1000,
                                      boost::bind(callback, _1, boost::ref(VITESSE_ROBOT), boost::ref(alpha),
                                                  boost::ref(target_linear_speed), boost::ref(target_angular_speed)));

    while (ros::ok()) {
        // Update speed at every loop
        current_linear_speed = current_linear_speed + alpha * (target_linear_speed - current_linear_speed);
        current_angular_speed = current_angular_speed + alpha * (target_angular_speed - current_angular_speed);
        // Avoid infinite convergence
        if (std::abs(current_linear_speed - target_linear_speed) < alpha * VITESSE_ROBOT) {
            current_linear_speed = target_linear_speed;
        }
        if (std::abs(current_linear_speed - target_linear_speed) < alpha * VITESSE_ROBOT) {
            current_angular_speed = target_angular_speed;
        }

        geometry_msgs::Twist vel_msg;
        vel_msg.linear.x = current_linear_speed;
        vel_msg.angular.z = current_angular_speed;
        pub.publish(vel_msg);
        // ROS_INFO("Linear speed = %f", current_linear_speed);
        // ROS_INFO("Angular speed = %f", current_angular_speed);

        ros::spinOnce();
        loop_rate.sleep();
    }

    return 0;
}
