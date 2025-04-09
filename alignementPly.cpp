#include <pcl/visualization/cloud_viewer.h>
#include <pcl/filters/voxel_grid.h>
#include <iostream>
#include <windows.h>
#include <string>
#include <filesystem>
#include <pcl/common/io.h>
#include <pcl/io/ply_io.h>
#include <pcl/common/transforms.h>
#include <pcl/memory.h>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/registration/icp.h>
#include <pcl/features/fpfh.h>
#include <pcl/features/pfh.h>
#include <pcl/features/normal_3d.h>
#include <pcl/registration/correspondence_estimation.h>
#include <pcl/registration/correspondence_rejection_sample_consensus.h>
#include <pcl/registration/ia_ransac.h>
#include <pcl/features/normal_3d_omp.h> 
#include <pcl/features/fpfh_omp.h>       
#include <future>  
#include <pcl/registration/icp_nl.h>
#include <omp.h>
#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>
#include <pcl/kdtree/kdtree_flann.h>
using namespace std;
using namespace pcl;
PointCloud<PointXYZ>::Ptr cloud(new PointCloud<PointXYZ>);
PointCloud<PointXYZ>::Ptr cloudsec(new PointCloud<PointXYZ>);
PointCloud<PointXYZ>::Ptr cloudres(new PointCloud<PointXYZ>);
PointCloud<PointXYZRGB>::Ptr cloud_rgb(new PointCloud<PointXYZRGB>);
pcl::PointCloud<pcl::Normal>::Ptr normalsply(new PointCloud<pcl::Normal>);
pcl::PointCloud<pcl::PointNormal>::Ptr cloud_normals(new PointCloud<pcl::PointNormal>);

using namespace std;
//function to load two pcd
void ReadPCloud()
{

    if (pcl::io::loadPCDFile<pcl::PointXYZ>("D:/project/alignementPly/point_cloud_pbr_1_subsub.pcd", *cloud) == -1) {
        PCL_ERROR("Impossible de lire le fichier PCD\n");
       
    }

    if (pcl::io::loadPCDFile<pcl::PointXYZ>("D:/project/alignementPly/pointcloud.pcd", *cloudsec) == -1) {
        PCL_ERROR("Impossible de lire le fichier PCD\n");
        
    }

    if (pcl::io::loadPLYFile<pcl::PointNormal>("D:/project/alignementPly/point_cloud_pbr_3.ply", *cloud_normals ) == -1) {
        PCL_ERROR("Erreur : impossible de charger le fichier PCD !\n");
        
    }

}

pcl::PointCloud<pcl::PointXYZRGB>::Ptr convertToPointXYZRGB(pcl::PointCloud<pcl::PointXYZ>::Ptr cl, uint8_t r = 255, uint8_t g = 255, uint8_t b = 255) {
    PointCloud<PointXYZRGB>::Ptr cloud_rgb(new PointCloud<PointXYZRGB>);

    // Parcourir chaque point et copier les coordonnées avec la couleur spécifiée
    for (const auto& point : cl->points) {
        pcl::PointXYZRGB point_rgb;
        point_rgb.x = point.x;
        point_rgb.y = point.y;
        point_rgb.z = point.z;
        point_rgb.r = r;
        point_rgb.g = g;
        point_rgb.b = b;
        cloud_rgb->points.push_back(point_rgb);
    }

    cloud_rgb->width = cl->width;
    cloud_rgb->height = cl->height;
    cloud_rgb->is_dense = cl->is_dense;

    return cloud_rgb;
}


void
loadingPlyAndPcdView()
{
  
    //création du visualizer
    pcl::visualization::PCLVisualizer::Ptr viewer(new pcl::visualization::PCLVisualizer("3D Viewer"));
    //chargement du point cloud
    cloud_rgb=convertToPointXYZRGB(cloudsec, 255, 0, 0);
    viewer->setBackgroundColor(0, 0, 0);
    viewer->addPointCloud(cloud, "cl", 0);
    //viewer->addPointCloud<pcl::PointNormal>(cloud_normals, "normals");
    viewer->addPointCloud(cloud_rgb, "cl2", 0);
    viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 5.0, "cl");
    viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 5.0, "cl2");
    //intialisation du viewer
    viewer->addCoordinateSystem(1.0);
    viewer->initCameraParameters();
    viewer->setCameraPosition(0, 0, 0, 0, 0, 0, 0);
    while (!viewer->wasStopped()) {
        viewer->spinOnce(100);
    }
}


pcl::PointCloud<pcl::Normal>::Ptr estimate_normals(
    pcl::PointCloud<pcl::PointXYZ>::Ptr cl,
    double radius
) {
    pcl::NormalEstimationOMP<pcl::PointXYZ, pcl::Normal> ne;  // Utilisation d'OpenMP
    ne.setNumberOfThreads(12);  // Nombre de threads pour le calcul parallèle

    pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>());
    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>());

    ne.setInputCloud(cl);
    ne.setSearchMethod(tree);
    ne.setRadiusSearch(radius);
    ne.compute(*normals);

    return normals;
}

pcl::PointCloud<pcl::Normal>::Ptr loadNormals(pcl::PointCloud<pcl::PointXYZ>::Ptr cl)
{
    pcl::PointCloud<pcl::Normal>::Ptr normals(new PointCloud<pcl::Normal>);
    return normals;
}


pcl::PointCloud<pcl::FPFHSignature33>::Ptr compute_fpfh_features(
    pcl::PointCloud<pcl::PointXYZ>::Ptr cl,
    pcl::PointCloud<pcl::Normal>::Ptr normals,
    double radius
) {
    pcl::FPFHEstimationOMP<pcl::PointXYZ, pcl::Normal, pcl::FPFHSignature33> fpfh;  // Utilisation d'OpenMP
    fpfh.setNumberOfThreads(12);

    pcl::PointCloud<pcl::FPFHSignature33>::Ptr features(new pcl::PointCloud<pcl::FPFHSignature33>());
    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>());

    fpfh.setInputCloud(cl);
    fpfh.setInputNormals(normals);
    fpfh.setSearchMethod(tree);
    fpfh.setRadiusSearch(radius);
    fpfh.compute(*features);

    return features;
}






pcl::PointCloud<pcl::PointXYZ>::Ptr align_with_descriptors(
    pcl::PointCloud<pcl::PointXYZ>::Ptr source_cloud,
    pcl::PointCloud<pcl::FPFHSignature33>::Ptr source_features,
    pcl::PointCloud<pcl::PointXYZ>::Ptr target_cloud,
    pcl::PointCloud<pcl::FPFHSignature33>::Ptr target_features,
    double distanceSamplemin,
    double distanceCorres,
    double IterMax
) {
    std::cout << "on commence align" << std::endl;

    pcl::SampleConsensusInitialAlignment<pcl::PointXYZ, pcl::PointXYZ, pcl::FPFHSignature33> sac_ia;

    sac_ia.setInputSource(source_cloud);
    sac_ia.setSourceFeatures(source_features);
    sac_ia.setInputTarget(target_cloud);
    sac_ia.setTargetFeatures(target_features);
    sac_ia.setMinSampleDistance(distanceSamplemin);
    sac_ia.setMaxCorrespondenceDistance(distanceCorres);
    sac_ia.setMaximumIterations(IterMax);

    pcl::PointCloud<pcl::PointXYZ>::Ptr aligned_cloud(new pcl::PointCloud<pcl::PointXYZ>());

    sac_ia.align(*aligned_cloud);
    

    std::cout << "on fini align" << std::endl;
    if (sac_ia.hasConverged()) {
        std::cout << "SAC-IA converged with score: " << sac_ia.getFitnessScore() << std::endl;
        std::cout << "SAC-IA matrice de transformation: " << sac_ia.getFinalTransformation() << std::endl;
    }
    else {
        std::cerr << "SAC-IA did not converge!" << std::endl;
    }

    return aligned_cloud;
}


void MergePC()
{
    *cloudres = *cloud + *cloudsec;

    pcl::VoxelGrid<pcl::PointXYZ> voxel_filter;
    voxel_filter.setInputCloud(cloudres);
    voxel_filter.setLeafSize(0.01f, 0.01f, 0.01f);
    pcl::PointCloud<pcl::PointXYZ>::Ptr filtered_cloud(new pcl::PointCloud<pcl::PointXYZ>);
    voxel_filter.filter(*filtered_cloud);
    *cloudres = *filtered_cloud;
    
    pcl::io::savePCDFileASCII("final_merged_cloud.pcd", *cloudres);
}



// Fonction pour appliquer une régularisation laplacienne sur les déplacements
void regularizeDeformation(PointCloud<pcl::PointXYZ>::Ptr deformed_cloud, PointCloud<pcl::PointXYZ>::Ptr original_cloud, float lambda = 0.1)
{
    pcl::KdTreeFLANN<PointXYZ> kdtree;
    kdtree.setInputCloud(original_cloud);

    for (size_t i = 0; i < deformed_cloud->size(); ++i) {
        PointXYZ& p = deformed_cloud->points[i];

        std::vector<int> pointIdx;
        std::vector<float> pointDist;
        if (kdtree.nearestKSearch(p, 5, pointIdx, pointDist) > 0) {
            Eigen::Vector3f avg(0, 0, 0);
            for (int idx : pointIdx) {
                avg += original_cloud->points[idx].getVector3fMap();
            }
            avg /= pointIdx.size();

            // Appliquer une correction lissée
            Eigen::Vector3f p_vec = p.getVector3fMap();
            p_vec = (1 - lambda) * p_vec + lambda * avg;
            p.getVector3fMap() = p_vec;
        }
    }
}


PointCloud<pcl::PointXYZ>::Ptr align_point_clouds(
    PointCloud<pcl::PointXYZ>::Ptr cloud_source,
    PointCloud<pcl::PointXYZ>::Ptr cloud_target,
    double iteration,
    double transfoEpsi,
    double Fitness,
    double Maxcorres,
    double Iterransac,
    double outlier
) {
    // Initialiser l'algorithme ICP
    pcl::IterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ> icp;
    icp.setInputSource(cloud_source);
    icp.setInputTarget(cloud_target);

    // Modifier les paramètres ICP
    icp.setMaximumIterations(iteration);
    icp.setTransformationEpsilon(transfoEpsi);
    icp.setEuclideanFitnessEpsilon(Fitness);
    icp.setMaxCorrespondenceDistance(Maxcorres);
    icp.setRANSACIterations(Iterransac);
    icp.setRANSACOutlierRejectionThreshold(outlier);

    // Nuage de points pour stocker le résultat aligné
    PointCloud<pcl::PointXYZ>::Ptr aligned_cloud(new PointCloud<pcl::PointXYZ>);

    // Exécuter l'alignement ICP en activant le multi-threading
#pragma omp parallel
    {
#pragma omp single
        std::cout << "Nombre de threads OpenMP utilisés : " << omp_get_num_threads() << std::endl;

#pragma omp single
        icp.align(*aligned_cloud);
    }

    // Vérifier la convergence
    if (icp.hasConverged()) {
        std::cout << "ICP converged with score: " << icp.getFitnessScore() << std::endl;
        std::cout << "Matrice de transformation: " << icp.getFinalTransformation() << std::endl;
    }
    else {
        std::cerr << "ICP did not converge!" << std::endl;
    }

    return aligned_cloud;
}


//----------------------
void regularizeWithNormals(pcl::PointCloud<pcl::PointNormal>::Ptr cloud, float lambda = 0.1) {
    pcl::KdTreeFLANN<pcl::PointNormal> kdtree;
    kdtree.setInputCloud(cloud);

    pcl::PointCloud<pcl::PointNormal>::Ptr smoothed(new pcl::PointCloud<pcl::PointNormal>);

    for (size_t i = 0; i < cloud->size(); ++i) {
        pcl::PointNormal p = cloud->points[i];

        std::vector<int> pointIdx;
        std::vector<float> pointDist;
        if (kdtree.nearestKSearch(p, 10, pointIdx, pointDist) > 0) { // 10 voisins

            Eigen::Vector3f avg_pos(0, 0, 0);
            Eigen::Vector3f avg_norm(0, 0, 0);
            float total_weight = 0.0;

            for (int idx : pointIdx) {
                float weight = exp(-pointDist[idx] / 0.01); // Poids gaussien
                avg_pos += weight * cloud->points[idx].getVector3fMap();
                avg_norm += weight * cloud->points[idx].getNormalVector3fMap();
                total_weight += weight;
            }

            avg_pos /= total_weight;
            avg_norm /= total_weight;
            avg_norm.normalize(); // On garde des normales unitaires

            p.getVector3fMap() = (1 - lambda) * p.getVector3fMap() + lambda * avg_pos;
            p.getNormalVector3fMap() = (1 - lambda) * p.getNormalVector3fMap() + lambda * avg_norm;
        }

        smoothed->push_back(p);
    }

    *cloud = *smoothed; // Mettre à jour le nuage
}

//---------------------------------

int main()
{
    ReadPCloud();
    cout<<"enter the first radius"<< endl;
    int radius;
    cin >> radius ;
   auto normsource= estimate_normals(cloud, radius);
   auto normtarget = estimate_normals(cloudsec, radius);
   cout << "enter the second radius" << endl;
   int radius2;
   cin >> radius2;
   auto featsource = compute_fpfh_features(cloud,normsource,radius2);
   auto feattarget = compute_fpfh_features(cloudsec, normtarget, radius2);
   cout << "enter the sample distance" << endl;
   float distSam;
   cin >> distSam;
   cout << "enter the max distance for correspondance" << endl;
   float distCor;
   cin >> distCor;
   cout << "enter the number of iteration" << endl;
   float iter;
   cin >> iter;
   cloud = align_with_descriptors(cloud, featsource, cloudsec, feattarget, distSam, distCor, iter);
   cloud = align_point_clouds(cloud,cloudsec,2000,1e-8,1e-5,1,2000,1.4);
   regularizeDeformation(cloud,cloudsec,1.2);
  // regularizeWithNormals(cloud_normals, 1.2);

   loadingPlyAndPcdView();

   MergePC();
	return 0;
}
