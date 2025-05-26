/**
 * @file GraphUpdateQueue.h
 * @brief Header file for the GraphUpdateQueue class, which manages a thread-safe queue for graph updates.
 *
 * This file defines the `GraphUpdateQueue` class, which provides a thread-safe queue
 * for storing updates to the network traffic graph (nodes and edges).  It uses a
 * condition variable to allow consumer threads to wait for updates.
 */

#ifndef GRAPHUPDATEQUEUE_H
#define GRAPHUPDATEQUEUE_H

#include <queue>
#include <mutex>
#include <condition_variable>
#include <vector>
#include <memory> // Include for std::unique_ptr

#include "GraphEdge.h"
#include "GraphNode.h"

/**
 * @brief Represents an update to the network traffic graph.
 *
 * This struct encapsulates either a node update or an edge update.  It uses a weak pointer
 * to the updated graph element to avoid creating a strong dependency and potential
 * circular references.
 */
struct GraphUpdate {
    /**
     * @brief Enum indicating the type of graph update.
     */
    enum class Type {
        /**
       * @brief Represents the creation of a new graph node.
       */
        NODE_CREATE,
        /**
         * @brief Represents an update to a graph node.
         */
        NODE_UPDATE,
        /**
         * @brief Represents te creation of a graph edge.
         */
        EDGE_CREATE
    };

    /**
     * @brief The type of graph update (node or edge).
     */
    Type type;
    /**
     * @brief A weak pointer to the updated graph node.
     */
    std::weak_ptr<GraphNode> node;
    /**
     * @brief A weak pointer to the updated graph edge.
     */
    std::weak_ptr<GraphEdge> edge;
};

/**
 * @brief Thread-safe queue for storing and processing graph updates.
 *
 * The `GraphUpdateQueue` class provides a thread-safe mechanism for adding and retrieving
 * updates to the network traffic graph.  It uses a mutex to protect the queue and a
 * condition variable to signal when new updates are available.
 */
class GraphUpdateQueue {
private:
    /**
     * @brief The queue storing the graph updates.
     */
    std::queue<GraphUpdate> updates;
    /**
     * @brief Mutex to protect access to the queue.
     */
    mutable std::mutex queue_mutex;
    /**
     * @brief Condition variable used to signal when new updates are available.
     */
    std::condition_variable cv;
    /**
     * @brief Flag indicating whether the queue is shutting down.
     */
    bool shutdown_flag = false;

public:
    /**
     * @brief Pushes a new graph update onto the queue.
     *
     * This method adds a new `GraphUpdate` to the queue and notifies any waiting threads
     * that an update is available.
     *
     * @param update The graph update to add to the queue.
     */
    void push(const GraphUpdate &update) {
        std::lock_guard<std::mutex> lock(queue_mutex);
        updates.push(update);
        cv.notify_one();
    }

    /**
     * @brief Pops a graph update from the queue.
     *
     * This method retrieves the next `GraphUpdate` from the queue.  If the queue is empty,
     * it waits until an update is available or the queue is shut down.
     *
     * @param update A reference to the GraphUpdate object where the retrieved update will be stored.
     * @return true if an update was successfully retrieved, false if the queue is shut down and empty.
     */
    bool pop(GraphUpdate &update) {
        std::unique_lock<std::mutex> lock(queue_mutex);
        cv.wait(lock, [this]() { return !updates.empty() || shutdown_flag; });

        if (shutdown_flag && updates.empty()) return false;

        update = updates.front();
        updates.pop();
        return true;
    }

    /**
     * @brief Pops all graph updates from the queue.
     *
     * This method retrieves all `GraphUpdate`s from the queue and returns them as a vector.
     * If the queue is empty, it returns an empty vector.  The queue is emptied
     * as a result of calling this method.
     *
     * @return A vector containing all graph updates currently in the queue.
     */
    std::vector<GraphUpdate> popAll() {
        std::lock_guard<std::mutex> lock(queue_mutex);

        // Create a temporary queue to swap
        std::queue<GraphUpdate> temp;
        std::swap(temp, updates); // Efficiently clears the main queue

        // Transfer elements to the vector
        std::vector<GraphUpdate> allUpdates;
        while (!temp.empty()) {
            allUpdates.push_back(std::move(temp.front())); // Move instead of copy
            temp.pop();
        }

        return std::move(allUpdates);
    }

    /**
     * @brief Shuts down the queue.
     *
     * This method sets the shutdown flag, preventing further updates from being added
     * and signals all waiting threads to wake up.  This allows threads waiting on
     * the queue to exit gracefully.
     */
    void shutdown() {
        std::lock_guard<std::mutex> lock(queue_mutex);
        shutdown_flag = true;
        cv.notify_all();
    }
};

#endif //GRAPHUPDATEQUEUE_H
