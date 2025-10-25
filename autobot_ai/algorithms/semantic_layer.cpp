#include <autobot_ai/semantic_layer.h>
#include <pluginlib/class_list_macros.h>

namespace costmap_2d {

SemanticLayer::SemanticLayer() : new_data_(false) {}

void SemanticLayer::onInitialize() {
    ros::NodeHandle nh("~/" + name_);
    semantic_sub_ = nh.subscribe("/semantic_costmap", 1, &SemanticLayer::semanticCallback, this);
    current_ = true;
}

void SemanticLayer::updateBounds(double robot_x, double robot_y, double robot_yaw, 
                              double* min_x, double* min_y, double* max_x, double* max_y) {
    if (!new_data_) return;
    
    *min_x = std::min(*min_x, latest_semantic_.info.origin.position.x);
    *min_y = std::min(*min_y, latest_semantic_.info.origin.position.y);
    *max_x = std::max(*max_x, latest_semantic_.info.origin.position.x + 
                     latest_semantic_.info.width * latest_semantic_.info.resolution);
    *max_y = std::max(*max_y, latest_semantic_.info.origin.position.y + 
                     latest_semantic_.info.height * latest_semantic_.info.resolution);
}

void SemanticLayer::updateCosts(costmap_2d::Costmap2D& master_grid, 
                             int min_i, int min_j, int max_i, int max_j) {
    if (!new_data_) return;
    
    unsigned int mx, my;
    for (size_t y = 0; y < latest_semantic_.info.height; ++y) {
        for (size_t x = 0; x < latest_semantic_.info.width; ++x) {
            double wx = latest_semantic_.info.origin.position.x + (x + 0.5) * latest_semantic_.info.resolution;
            double wy = latest_semantic_.info.origin.position.y + (y + 0.5) * latest_semantic_.info.resolution;
            
            if (master_grid.worldToMap(wx, wy, mx, my)) {
                int cost = latest_semantic_.data[y * latest_semantic_.info.width + x];
                if (cost != -1) {  // Only update known cells
                    master_grid.setCost(mx, my, cost);
                }
            }
        }
    }
    new_data_ = false;
}

void SemanticLayer::semanticCallback(const nav_msgs::OccupancyGridConstPtr& msg) {
    latest_semantic_ = *msg;
    new_data_ = true;
}

PLUGINLIB_EXPORT_CLASS(costmap_2d::SemanticLayer, costmap_2d::Layer)

}  // namespace costmap_2d