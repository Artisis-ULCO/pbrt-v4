#ifndef _INTERPOLATION_HPP
#define _INTERPOLATION_HPP
/** @file Interpolation.hpp.
 * Interpolation de n Point3f ou n Vector3f. On utilise des B-Splines
 * et des splines de Catmull-Rom.
 * Classe originellement développée pour Igloo et réadaptée par C. Renaud
 * pour Pbrt - juin 2020
 * @author Michel Leblond
 * @version 2.0
 * @date mai 2013
 */

#include <pbrt/util/vecmath.h>
#include <vector>

namespace pbrt {
  
class CInterpolation {
public:
  /** procédure d'interpolation.
   */ 
  //  static void DeBoor(int n, int k, int nb, CPoint * P, CPoint * Q);
  static void Interpol(int n ,int nb , double *P, double *Q);
  // méthodes d'interpolation basées sur CatmullRom. Les paramètres
  // correspondent à :
  // - nb : le nombre de valeurs à générer par interpolation
  // - P : le vecteur des points de contrôle
  // - Q : le vecteur contenant les valeurs générées
  /*static void CatmullRom(int n, int nb,
			 std::vector<Point3f> &P,
			 std::vector<Point3f> &Q);
  static void CatmullRom(int n, int nb,
			 std::vector<Vector3f> &P,
			 std::vector<Vector3f> &Q);*/
  static void CatmullRom(int n, int nb, double *P, double *Q);
  
};//Interpolation

}
#endif
