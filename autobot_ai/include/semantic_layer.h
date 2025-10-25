#ifndef SEMANTIC_LAYER_H
#define SEMANTIC_LAYER_H

#include <costmap_2d/layer.h>
#include <costmap_2d/layered_costmap.h>
#include <nav_msgs/OccupancyGrid.h>
#include <ros/ros.h>

namespace costmap_2d {

class SemanticLayer : public Layer {
public:
    SemanticLayer();

    virtual void onInitialize();
    virtual void updateBounds(double robot_x, double robot_y, double robot_yaw, 
                            double* min_x, double* min_y, double* max_x, double* max_y);
    virtual void updateCosts(costmap_2d::Costmap2D& master_grid, 
                           int min_i, int min_j, int max_i, int max_j);
    
private:
    void semanticCallback(const nav_msgs::OccupancyGridConstPtr& msg);

    ros::Subscriber semantic_sub_;
    nav_msgs::OccupancyGrid latest_semantic_;
    bool new_data_;
};

}  // namespace costmap_2d

#endif