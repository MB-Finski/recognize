<?php
/*
 * Copyright (c) 2022 The Recognize contributors.
 * This file is licensed under the Affero General Public License version 3 or later. See the COPYING file.
 */

namespace OCA\Recognize\Service;

use OCA\Recognize\Clustering\HDBSCAN;
use OCA\Recognize\Db\FaceCluster;
use OCA\Recognize\Db\FaceClusterMapper;
use OCA\Recognize\Db\FaceDetection;
use OCA\Recognize\Db\FaceDetectionMapper;
use \Rubix\ML\Datasets\Labeled;
use \Rubix\ML\Kernels\Distance\Euclidean;

class FaceClusterAnalyzer {
        public const MIN_DATASET_SIZE = 120;
        public const MIN_DETECTION_SIZE = 0.03;
        public const MIN_CLUSTER_SEPARATION = 0.0;
        public const MAX_CLUSTER_EDGE_LENGTH = 0.5;
        public const DIMENSIONS = 128;
        public const SAMPLE_SIZE_EXISTING_CLUSTERS = 80;
        public const MAX_OVERLAP_NEW_CLUSTER = 0.1;
        public const MIN_OVERLAP_EXISTING_CLUSTER = 0.5;

        private FaceDetectionMapper $faceDetections;
        private FaceClusterMapper $faceClusters;
        private Logger $logger;
        private int $minDatasetSize = self::MIN_DATASET_SIZE;

        public function __construct(FaceDetectionMapper $faceDetections, FaceClusterMapper $faceClusters, Logger $logger) {
                $this->faceDetections = $faceDetections;
                $this->faceClusters = $faceClusters;
                $this->logger = $logger;
        }

        public function setMinDatasetSize(int $minSize) : void {
                $this->minDatasetSize = $minSize;
        }

        /**
         * @throws \OCP\DB\Exception
         * @throws \JsonException
         */
        public function calculateClusters(string $userId, int $batchSize = 0): void {
                $this->logger->debug('ClusterDebug: Retrieving face detections for user ' . $userId);

                if ($batchSize === 0) {
                        ini_set('memory_limit', -1);
                }

                $unclusteredDetections = $this->faceDetections->findUnclusteredByUserId($userId, $batchSize);

                $unclusteredDetections = array_values(array_filter($unclusteredDetections, fn ($detection) =>
                        $detection->getHeight() > self::MIN_DETECTION_SIZE && $detection->getWidth() > self::MIN_DETECTION_SIZE
                ));

                #DEBUG:
                #$unclusteredDetections = array_slice($unclusteredDetections, 0, 4000);

                if (count($unclusteredDetections) < $this->minDatasetSize) {
                        $this->logger->debug('ClusterDebug: Not enough face detections found');
                        return;
                }

                $this->logger->debug('ClusterDebug: Found ' . count($unclusteredDetections) . " unclustered detections. Calculating clusters.");

                $sampledDetections = [];

                $existingClusters = $this->faceClusters->findByUserId($userId);
                $maxVotesByCluster = [];
                foreach ($existingClusters as $existingCluster) {
                        $sampled = $this->faceDetections->findClusterSample($existingCluster->getId(), self::SAMPLE_SIZE_EXISTING_CLUSTERS);
                        $sampledDetections = array_merge($sampledDetections, $sampled);
                        $maxVotesByCluster[$existingCluster->getId()] = count($sampled);
                }

                $detections = array_merge($unclusteredDetections, $sampledDetections);

                $dataset = new Labeled(array_map(static function (FaceDetection $detection): array {
                        return $detection->getVector();
                }, $detections), array_combine(array_keys($detections), array_keys($detections)), false);

                $n = count($detections);
                $hdbscan = new HDBSCAN($dataset, $this->getMinClusterSize($n), $this->getMinSampleSize($n));

                $numberOfClusteredDetections = 0;
                $clusters = $hdbscan->predict(self::MIN_CLUSTER_SEPARATION, self::MAX_CLUSTER_EDGE_LENGTH);

                foreach ($clusters as $flatCluster) {
                        $detectionKeys = array_keys($flatCluster->getClusterVertices());

                        $clusterDetections = array_filter($detections, function ($key) use ($detectionKeys) {
                                return isset($detectionKeys[$key]);
                        }, ARRAY_FILTER_USE_KEY);
                        $clusterCentroid = self::calculateCentroidOfDetections($clusterDetections);

                        /**
                         * @var int|false $detection
                         */
                        #$detection = current(array_filter($detectionKeys, static fn ($key) => $detections[$key]->getClusterId() !== null));
                        #votes = array_count_values(
                        $votes = [];

                        foreach ($detectionKeys as $detectionKey) {

                                if ($detectionKey < count($unclusteredDetections)) {
                                        continue;
                                }

                                $vote = $detections[$detectionKey]->getClusterId();

                                if($vote === null) {
                                        $vote = -1;
                                }

                                $votes[] = $vote;
                        }

                        if (empty($votes)) {
                                $overlap = 0.0;
                        } else {
                                $votes = array_count_values($votes);
                                $oldClusterId = array_search(max($votes), $votes);
                                $overlap = max($votes) / $maxVotesByCluster[$oldClusterId];
                        }

                        if ($overlap > self::MIN_OVERLAP_EXISTING_CLUSTER) {
                                $clusterId = $oldClusterId;
                                $cluster = $this->faceClusters->find($clusterId);
                        } elseif ($overlap < self::MAX_OVERLAP_NEW_CLUSTER) {

                                #$distance = new Euclidean();
                                #$distances = array_map(function ($detection) use ($clusterCentroid, $distance) {
                                #       return $distance->compute($detection->getVector(), $clusterCentroid);
                                #}, $clusterDetections);
                                #$clusterRadius = max($distances);

                                #if ($clusterRadius > 0.65) {
                                #       $this->logger->debug('ClusterDebug: Inner cluster distance for a cluster is ' . $clusterRadius . ' which is too large. Ignoring cluster.');
                                #       continue;
                                #}

                                $cluster = new FaceCluster();
                                $cluster->setTitle('');
                                $cluster->setUserId($userId);
                                $this->faceClusters->insert($cluster);
                        } else {
                                continue;
                        }

                        foreach ($detectionKeys as $detectionKey) {
                                if ($detectionKey >= count($unclusteredDetections)) {
                                        // This is a sampled, already clustered detection, ignore.
                                        continue;
                                }

                                // If threshold is larger than 0 and $clusterCentroid is not the null vector
                                if ($unclusteredDetections[$detectionKey]->getThreshold() > 0.0 && count(array_filter($clusterCentroid, fn ($el) => $el !== 0.0)) > 0) {
                                        // If a threshold is set for this detection and its vector is farther away from the centroid
                                        // than the threshold, skip assigning this detection to the cluster
                                        $distanceValue = self::distance($clusterCentroid, $unclusteredDetections[$detectionKey]->getVector());
                                        if ($distanceValue >= $unclusteredDetections[$detectionKey]->getThreshold()) {
                                                continue;
                                        }
                                }

                                $this->faceDetections->assocWithCluster($unclusteredDetections[$detectionKey], $cluster);
                                $numberOfClusteredDetections += 1;
                        }
                }

                $this->logger->debug('ClusterDebug: Clustering complete. Total num of clustered detections: ' . $numberOfClusteredDetections);
                $this->pruneClusters($userId);
        }

        /**
         * @throws \OCP\DB\Exception
         */
        public function pruneClusters(string $userId): void {
                $clusters = $this->faceClusters->findByUserId($userId);

                if (count($clusters) === 0) {
                        $this->logger->debug('No face clusters found');
                        return;
                }

                foreach ($clusters as $cluster) {
                        $detections = $this->faceDetections->findByClusterId($cluster->getId());

                        $filesWithDuplicateFaces = $this->findFilesWithDuplicateFaces($detections);
                        if (count($filesWithDuplicateFaces) === 0) {
                                continue;
                        }

                        $centroid = self::calculateCentroidOfDetections($detections);

                        foreach ($filesWithDuplicateFaces as $fileDetections) {
                                $detectionsByDistance = [];
                                foreach ($fileDetections as $detection) {
                                        $distance = new Euclidean();
                                        $detectionsByDistance[$detection->getId()] = $distance->compute($centroid, $detection->getVector());
                                }
                                asort($detectionsByDistance);
                                $bestMatchingDetectionId = array_keys($detectionsByDistance)[0];

                                foreach ($fileDetections as $detection) {
                                        if ($detection->getId() === $bestMatchingDetectionId) {
                                                continue;
                                        }
                                        $detection->setClusterId(null);
                                        $this->faceDetections->update($detection);
                                }
                        }
                }
        }

        /**
         * @param FaceDetection[] $detections
         * @return list<float>
         */
        public static function calculateCentroidOfDetections(array $detections): array {
                // init 128 dimensional vector
                /** @var list<float> $sum */
                $sum = [];
                for ($i = 0; $i < self::DIMENSIONS; $i++) {
                        $sum[] = 0;
                }

                if (count($detections) === 0) {
                        return $sum;
                }

                foreach ($detections as $detection) {
                        $sum = array_map(static function ($el, $el2) {
                                return $el + $el2;
                        }, $detection->getVector(), $sum);
                }

                $centroid = array_map(static function ($el) use ($detections) {
                        return $el / count($detections);
                }, $sum);

                return $centroid;
        }

        /**
         * @param array<FaceDetection> $detections
         * @return array<int,FaceDetection[]>
         */
        private function findFilesWithDuplicateFaces(array $detections): array {
                $files = [];
                foreach ($detections as $detection) {
                        if (!isset($files[$detection->getFileId()])) {
                                $files[$detection->getFileId()] = [];
                        }
                        $files[$detection->getFileId()][] = $detection;
                }

                /** @var array<int,FaceDetection[]> $filesWithDuplicateFaces */
                $filesWithDuplicateFaces = array_filter($files, static function ($detections) {
                        return count($detections) > 1;
                });

                return $filesWithDuplicateFaces;
        }

        private static ?Euclidean $distance;

        /**
         * @param list<int|float> $v1
         * @param list<int|float> $v2
         * @return float
         */
        private static function distance(array $v1, array $v2): float {
                if (!isset(self::$distance)) {
                        self::$distance = new Euclidean();
                }
                return self::$distance->compute($v1, $v2);
        }

        private function getMinClusterSize(int $n) : int {
                return (int)round(max(2, min(8, $n ** (1 / 4))));
        }

        private function getMinSampleSize(int $n) : int {
                return (int)round(max(2, min(3, $n ** (1 / 4))));
        }
}
