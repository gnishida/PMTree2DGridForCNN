﻿#include "PMTree2D.h"
#include "Utils.h"
#include <sstream>
#include <iostream>
#include "RenderManager.h"
#include "Camera.h"

namespace pmtree {

	const float M_PI = 3.1415926535f;
	const int NUM_SEGMENTS = 30;
	const int NUM_LEVELS = 3;
	const float MIN_SEGMENT_WIDTH = 0.005f;

	/**
	* Shape ratioを返却する。
	* 論文Cretion and rendering of realistic treesの4.3節に記載されている内容に基づく。
	*
	* @param shape		shape id
	* @param ratio		ratio
	* @return			shape ratio
	*/
	float shapeRatio(int shape, float ratio) {
		if (shape == 0) {
			return 0.2f + 0.8f * ratio;
		}
		else if (shape == 1) {
			return 0.2f + 0.8f * sinf(M_PI * ratio);
		}
		else if (shape == 2) {
			return 0.2f + 0.8f * sinf(0.5f * M_PI * ratio);
		}
		else if (shape == 3) {
			return 1.0f;
		}
		else if (shape == 4) {
			return 0.5f + 0.5f * ratio;
		}
		else if (shape == 5) {
			if (ratio <= 0.7f) {
				return ratio / 0.7f;
			}
			else {
				return (1.0f - ratio) / 0.3f;
			}
		}
		else if (shape == 6) {
			return 1.0f - 0.8f * ratio;
		}
		else if (shape == 7) {
			if (ratio <= 0.7f) {
				return 0.5f + 0.5f * ratio / 0.7f;
			}
			else {
				return 0.5f + 0.5f * (1.0f - ratio) / 0.3f;
			}
		}
		else {
			return 0.0f;
		}
	}

	TreeNode::TreeNode(boost::shared_ptr<TreeNode> parent, int level, int index, float segmentLength, float attenuationFactor, float baseFactor, float curve, float curveBack) {
		this->parent = parent;
		this->level = level;
		this->index = index;
		this->segmentLength = segmentLength;
		this->baseFactor = baseFactor;
		this->curve = curve;
		this->curveBack = curveBack;
		this->attenuationFactor = attenuationFactor;

		attenuationFactor = 0.0f;
		curveV = 0.0f;
		rotateV = 0.0f;
	}

	void TreeNode::generateRandom() {
		if (level == 0 && index == 0) {
			baseFactor = utils::uniform(0.0f, 0.5f);
		}
		
		if (index == 0) {
			curve = utils::uniform(-90, 90);
			curveBack = utils::uniform(-90, 90);
			if (level > 0) {
				curveV = utils::uniform(-90, 90);
			}
		}
		else {
			if (index < NUM_SEGMENTS / 2.0f) {
				curveV = utils::uniform(-5, 5) + curve / NUM_SEGMENTS / 2.0f;
			}
			else {
				curveV = utils::uniform(-5, 5) + curveBack / NUM_SEGMENTS / 2.0f;
			}
		}

		rotateV = 59.0f;
	}

	std::string TreeNode::to_string() {
		std::stringstream ss;
		
		ss << baseFactor << "," << attenuationFactor << "," << (curve + 90) / 180.0f;

		return ss.str();
	}

	void TreeNode::recover(const std::vector<float>& params) {
		/*
		attenuationFactor = params[0];
		curve = params[1] * 180.0f - 90.0f;
		*/
	}

	PMTree2D::PMTree2D() {
		root = boost::shared_ptr<TreeNode>(new TreeNode(NULL, 0, 0, 0, 0, 0, 0, 0));
	}

	void PMTree2D::generateRandom() {
		root = boost::shared_ptr<TreeNode>(new TreeNode(NULL, 0, 0, 10.0f / NUM_SEGMENTS, 1.0f, 0.0f, 0, 0));
		root->generateRandom();

		std::list<boost::shared_ptr<TreeNode> > queue;
		queue.push_back(root);

		// generate random param values for branches in the breadth-first order
		while (!queue.empty()) {
			boost::shared_ptr<TreeNode> node = queue.front();
			queue.pop_front();

			if (node->index < NUM_SEGMENTS - 1) {
				// extend the segment
				boost::shared_ptr<TreeNode> child = boost::shared_ptr<TreeNode>(new TreeNode(node, node->level, node->index + 1, node->segmentLength, 1.0f, node->baseFactor, node->curve, node->curveBack));
				child->generateRandom();
				node->children.push_back(child);
				queue.push_back(child);

				if (node->level < NUM_LEVELS) {
					if (node->level > 0 || node->index + 1 > NUM_SEGMENTS * node->baseFactor) {
						if (utils::uniform(0, 1) > 0.4f) {
							// branching
							float attenuationFactor;
							if (node->level == 0) {
								attenuationFactor = utils::uniform(0.5f, 0.8f) * shapeRatio(7, (NUM_SEGMENTS - node->index - 1) / (NUM_SEGMENTS * (1.0f - node->baseFactor)));
							}
							else {
								attenuationFactor = utils::uniform(0.3f, 0.6f) * (NUM_SEGMENTS - node->index * 0.9f) / NUM_SEGMENTS;
							}

							boost::shared_ptr<TreeNode> child = boost::shared_ptr<TreeNode>(new TreeNode(node, node->level + 1, 0, node->segmentLength, attenuationFactor, 0.0f, 0.0f, 0.0f));
							child->generateRandom();
							node->children.push_back(child);
							queue.push_back(child);
						}
					}
				}
			}
		}
	}

	bool PMTree2D::generateGeometry(RenderManager* renderManager, bool fixed_width) {
		bool underground = false;

		glm::mat4 modelMat;
		float length = 10.0f / NUM_SEGMENTS;
		float width = 0.3f;
		if (fixed_width) {
			width = 0.03f;
		}

		std::vector<Vertex> vertices;
		if (generateSegmentGeometry(renderManager, modelMat, length, width, fixed_width, root, vertices)) underground = true;
		renderManager->addObject("tree", "", vertices, true);

		return underground;
	}

	bool PMTree2D::generateSegmentGeometry(RenderManager* renderManager, const glm::mat4& modelMat, float segment_length, float segment_width, bool fixed_width, boost::shared_ptr<TreeNode>& node, std::vector<Vertex>& vertices) {
		bool underground = false;

		glm::mat4 mat = modelMat;

		mat = glm::rotate(mat, node->rotateV / 180.0f * M_PI, glm::vec3(0, 1, 0));
		mat = glm::rotate(mat, node->curveV / 180.0f * M_PI, glm::vec3(0, 0, 1));

		float w1 = segment_width;
		if (!fixed_width) {
			w1 = (segment_width - MIN_SEGMENT_WIDTH) * (NUM_SEGMENTS - node->index) / NUM_SEGMENTS + MIN_SEGMENT_WIDTH;
		}

		float w2 = segment_width;
		if (!fixed_width) {
			w2 = (segment_width - MIN_SEGMENT_WIDTH) * (NUM_SEGMENTS - node->index - 1) / NUM_SEGMENTS + MIN_SEGMENT_WIDTH;
		}

		glm::vec4 color(1, 0, 0, 1.0);
		if (node->level > 0) {
			color = glm::vec4(0, 1, 0, 1);
		}
		glutils::drawCylinderY(w1 * 0.5, w2 * 0.5, segment_length, color, mat, vertices);
		
		mat = glm::translate(mat, glm::vec3(0, segment_length, 0));

		if (node->children.size() >= 1) {
			// extend the segment
			generateSegmentGeometry(renderManager, mat, segment_length, segment_width, fixed_width, node->children[0], vertices);
		}
		
		if (node->children.size() >= 2) {
			if (node->level < NUM_LEVELS - 1) {
				// branching
				if (fixed_width) {
					generateSegmentGeometry(renderManager, mat, segment_length * node->children[1]->attenuationFactor, segment_width, fixed_width, node->children[1], vertices);
				}
				else {
					generateSegmentGeometry(renderManager, mat, segment_length * node->children[1]->attenuationFactor, std::max(MIN_SEGMENT_WIDTH, w1 * node->children[1]->attenuationFactor), fixed_width, node->children[1], vertices);
				}
			}
			else {
				generateLeafGeometry(renderManager, mat, segment_length * node->children[1]->attenuationFactor, node->children[1], vertices);
			}
		}

		return underground;
	}

	void PMTree2D::generateLeafGeometry(RenderManager* renderManager, const glm::mat4& modelMat, float segment_length, boost::shared_ptr<TreeNode>& node, std::vector<Vertex>& vertices) {
		glm::mat4 mat = modelMat;

		mat = glm::rotate(mat, node->rotateV / 180.0f * M_PI, glm::vec3(0, 1, 0));
		mat = glm::rotate(mat, 75.0f / 180.0f * M_PI, glm::vec3(0, 0, 1));

		float leaf_length = 0.1f;
		mat = glm::translate(mat, glm::vec3(0, leaf_length * 0.5, 0));

		glutils::drawCircle(leaf_length * 0.25, leaf_length * 0.5, glm::vec4(0, 0, 1, 1.0), mat, vertices);
	}

	void PMTree2D::generateTrainingData(const cv::Mat& image, Camera* camera, int screenWidth, int screenHeight, std::vector<cv::Mat>& localImages, std::vector<std::vector<float> >& parameters) {
		// 画像に空白マージンを付加する
		int padding = 300;
		cv::Mat imagePadded(image.rows + padding * 2, image.cols + padding * 2, image.type(), cv::Scalar(255));
		image.copyTo(imagePadded(cv::Rect(padding, padding, image.cols, image.rows)));
		//cv::imwrite("image_padded.jpg", imagePadded);

		generateTrainingData(glm::mat4(), 10.0f / NUM_SEGMENTS, root, imagePadded, padding, camera, screenWidth, screenHeight, localImages, parameters);
	}

	void PMTree2D::generateTrainingData(const glm::mat4& modelMat, float segment_length, boost::shared_ptr<TreeNode>& node, const cv::Mat& imagePadded, int padding, Camera* camera, int screenWidth, int screenHeight, std::vector<cv::Mat>& localImages, std::vector<std::vector<float> >& parameters) {
		// 座標系を回転
		glm::mat4 mat = modelMat;
		mat = glm::rotate(mat, node->curveV / 180.0f * M_PI, glm::vec3(0, 0, 1));

		// current positionを計算
		glm::vec4 p(0, segment_length, 0, 1);
		p = mat * p;
		p = camera->mvpMatrix * p;
		glm::vec2 pp((p.x / p.w + 1.0f) * 0.5f * screenWidth, screenHeight - (p.y / p.w + 1.0f) * 0.5f * screenHeight);

		// current positionからsegment_length下がった点を計算
		glm::vec4 p2(0, 0, 0, 1);
		p2 = mat * p2;
		p2 = camera->mvpMatrix * p2;
		glm::vec2 pp2((p2.x / p2.w + 1.0f) * 0.5f * screenWidth, screenHeight - (p2.y / p2.w + 1.0f) * 0.5f * screenHeight);

		// cropping sizeを計算
		//float crop_size = glm::length(pp2 - pp) * 5.0f;
		float crop_size = 64;

		if (crop_size <= 0) return;

		// matから、回転角度を抽出
		float theta = asinf(mat[0][1]);

		// 画像を回転
		cv::Mat rotatedImage;
		cv::Mat affineMatrix = cv::getRotationMatrix2D(cv::Point2d(pp.x + padding, pp.y + padding), -theta / M_PI * 180, 1.0);
		cv::warpAffine(imagePadded, rotatedImage, affineMatrix, imagePadded.size(), cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(255));

		//cv::imwrite("image_rotated.jpg", rotatedImage);

		// 画像をcropping
		cv::Rect roi(pp.x - crop_size / 2 + padding, pp.y - crop_size + padding, crop_size, crop_size);
		cv::Mat croppedImage = rotatedImage(roi);

		// 128x128にresize
		cv::resize(croppedImage, croppedImage, cv::Size(128, 128));
		cv::threshold(croppedImage, croppedImage, 200, 255, CV_THRESH_BINARY);
		//cv::imwrite("image_cropped.jpg", croppedImage);

		localImages.push_back(croppedImage);

		// パラメータを格納
		std::vector<float> params;
		if (node->children.size() >= 1) {
			params.push_back(1);
			params.push_back((node->children[0]->curveV + 90.0f) / 180.0f);
		}
		else {
			params.push_back(0);
			params.push_back(0.5);
		}
		if (node->children.size() >= 2) {
			params.push_back(1);
			params.push_back((node->children[1]->curveV + 90.0f) / 180.0f);
			//params.push_back(node->children[1]->attenuationFactor);
		}
		else {
			params.push_back(0);
			params.push_back(0.5);
			//params.push_back(0);
		}
		parameters.push_back(params);

		mat = glm::translate(mat, glm::vec3(0, segment_length, 0));

		// 子ノードの枝へ、再起処理
		if (node->children.size() >= 1) {
			generateTrainingData(mat, segment_length, node->children[0], imagePadded, padding, camera, screenWidth, screenHeight, localImages, parameters);

			if (node->level <= 1 && node->children.size() >= 2) {
				generateTrainingData(mat, segment_length * node->children[1]->attenuationFactor, node->children[1], imagePadded, padding, camera, screenWidth, screenHeight, localImages, parameters);
			}
		}
	}

	std::string PMTree2D::to_string() {
		std::stringstream ss;
		bool first_node = true;

		std::list<boost::shared_ptr<TreeNode> > queue;
		queue.push_back(root);

		while (!queue.empty()) {
			boost::shared_ptr<TreeNode> node = queue.front();
			queue.pop_front();
			
			if (first_node) {
				first_node = false;
			}
			else {
				ss << ",";
			}
			
			ss << node->to_string();

			for (int i = 0; i < node->children.size(); ++i) {
				queue.push_back(node->children[i]);
			}
		}

		return ss.str();
	}

	std::string PMTree2D::to_string(int index) {
		std::stringstream ss;
		bool first_node = true;

		std::list<boost::shared_ptr<TreeNode> > queue;
		queue.push_back(root);

		int count = 0;
		while (!queue.empty()) {
			boost::shared_ptr<TreeNode> node = queue.front();
			queue.pop_front();

			if (first_node) {
				first_node = false;
			}
			else {
				ss << ",";
			}

			ss << node->to_string();

			for (int i = 0; i < node->children.size(); ++i) {
				queue.push_back(node->children[i]);
			}

			count++;
			if (count >= index) break;
		}

		return ss.str();
	}

	void PMTree2D::recover(const std::vector<std::vector<float> >& params) {
		/*
		root = boost::shared_ptr<TreeNode>(new TreeNode(NULL, 0, 0));
		std::list<boost::shared_ptr<TreeNode> > queue;
		queue.push_back(root);

		int count = 0;
		while (!queue.empty()) {
			boost::shared_ptr<TreeNode> node = queue.front();
			queue.pop_front();

			node->recover(params[count++]);
			if (node->level < NUM_LEVELS - 1) {
				for (int k = 0; k < node->branching.size(); ++k) {
					boost::shared_ptr<TreeNode> child = boost::shared_ptr<TreeNode>(new TreeNode(node, node->level + 1, k));
					node->children.push_back(child);
					queue.push_back(child);
				}
			}
		}
		*/
	}
}