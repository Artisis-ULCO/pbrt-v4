/** @file Interpolation.cpp.
 * Interpolation de n Point3f ou Vector3f
 * Utilise des B-Splines ou les splines de Catmull-Rom
 */

#include <pbrt/interpolation.h>
#include <iostream>

namespace pbrt {
  
#ifdef SPLINES
/** création du tableau de noeuds.
 *@param n nombre de points de contréle
 *@param k degré des B-Splines
 *@param t tableau des noeuds de dimension n+k.
 *Les k premiers et les k derniers noeuds sont placés de 1 en 1
 *Les noeuds intermédiares sont placés en utilisant la distance
 *entre P[i-1] et P[i-k].
 */
void noeud (int n, int k, CPoint *P, double *t){
  t[0]=0.0;
  for(int i= 1;i<k;i++){
    t[i]=t[i-1]+1.0;
  }//for i
  for(int j=k;j<=n;j++){
    t[j]=t[j-1]+(P[j-1]-P[j-k]);
  }//for i
  for(int l=n+1;l<n+k;l++){
    t[l]=t[l-1]+1;
  }//for i
}//noeud

/** procédure d'interpolation de n CPoint
 * ou de n CVector (la classe CVector hérite de CPoint).
 *@param n nombre de points de contrôle
 *@param k degré des B-Splines
 *@param nb nombre de points souhaités de la courbe d'interpolation
 *@param P tableau des points de contrôle
 *@param Q tableau des nb points de la courbe d'interpolation
 */ 
void CInterpolation::DeBoor(int n, int k, int nb, std::vector <Point3f> &P, std::vector <Point3f> &Q){
  int nplusk = n+k;
  // il faut n+k noeuds
  int nbMax=nb*nb*10;
  // on va calculer nbMax points de la courbe
  CPoint *C;
  // pour les ranger
  double L=0.0;
  // longueur de l'arc de courbe
  double l=0.0;
  //pour totaliser la longueur d'un point Pi au point courant
  CPoint *d;
  // d vecteur temporaire des colonnes de l'algorithme de De Boor
  double *tnoeud;
  // vecteur des noeuds de l'algorithme de De Boor
  int i, j, h, ii, ik1;
  // variables de travail
  double pas, den, bg, bd, t,sca,scb;
  // variables de travail

  /****************/
  /*initialisation*/
  /****************/

  C = new CPoint[nbMax];
  tnoeud = new double [nplusk]; 
  d = new CPoint[k];

  noeud(n, k,P,tnoeud);
  t=tnoeud[k-1];
  //premier paramètre de la courbe
  pas=(tnoeud[n]-tnoeud[k-1])/(double)(nbMax-1);
  //incrément du paramètre

  /**********************/
  /* calcul du vecteur C :*/
  /**********************/

  /* calcul de nbMax points de la courbe et de
   * la longueur de l'arc de courbe
   */
  for(h=0;h<nbMax;h++) {
    //déterminer ii tel que tnoeud[ii]<=t<tnoeud[ii+1]
    ii=0;
    while((tnoeud[ii]<=t)&&(ii<n)){
      ii++;
    }
    //en sortie de boucle on a trouvé ii+1
    //calculer la première colonne du tableau triangulaire : i.e.
    //initialiser d avec les k points de contrôle
    //P[ii-k+1], P[ii-k+2], ..., P[ii]
    ik1 = ii-k;
    //en fait ii-k+1
    for(i=0;i<k;i++){
      d[i] = P[ik1+i];
    }//for i
    //calcul de C[h].
    //calculer dans d les colonnes d1 à d(k-1).
    for(j=1;j<k;j++) {
      for(i=0;i<k-j;i++) {
	bd=tnoeud[ii+i];
	bg=tnoeud[ik1+i+j];
	den=bd-bg;
	if(den!=0.0) {
	  sca=(bd-t)/den;
	  scb=(t-bg)/den;
	  d[i]=(d[i]*sca)+(d[i+1]*(scb));
	}
      }//for i
    }//for j
    C[h]=d[0];
    if (h!=0) {
      L +=C[h-1]-C[h];
    }
    t+=pas;
    //valeur suivante du paramètre t
  }//for h

  /***************/
  /* Calcul de Q */
  /***************/ 
	
  // les nb points de la courbe déterminent nb-1 intervalles de longueur
  // L/(nb-1)
  L/=(double)(nb-1);
  /* on choisit dans C nb points distants les uns des autres de L et
   * on les range dans Q
   */
  Q[0]=C[0];
  h=1;
  bool fini;
  for(i=1;i<nb;i++) {
    fini=false;
    l=0.0;
    while(!fini) {
      l+=C[h]-C[h-1];
      if(l>=L) {
	fini=true;
	if(l>L) {
	  h--;
	}
	Q[i]=C[h];
      }
      h++;
    }//while
  }//for
  delete [] tnoeud;
  delete [] C;
}//DeBoor

#endif

void CInterpolation::Interpol(int n ,int nb, double *P, double *Q) {
  double x=0.0, tmp, pas, pasCourant, s;
  int h;
  for(int l=0;l<n-1;l++) {
    tmp=P[l+1]-P[l];
    if(tmp<0.0) {
      tmp=-tmp;
    }
    x+=tmp;
  }//for
  Q[0]=P[0];
  h=0;
  pas=x/(double)(nb-1);
  if(P[1]-P[0]>0.0) {
    s=1.0;
  }else{
    s=-1.0;
  }//if
  pasCourant=pas*s;
  tmp=Q[0]+pasCourant;
  int i=1;
  h=0;
  while(h<nb-1){
    if(((s>0.0)&&(tmp>P[i]))||((s<0.0)&&(tmp<P[i]))) { 
      // la position P[i-1] a été dépassé.
      // on détermine le nouveau sens de parcours s
      i++; 
      if(P[i]-P[i-1]>0.0) {
	s=1.0;
      }else{
	s=-1.0;
      }//if
      // On ajoute ou on retranche la différence
      // selon le nouveau sens de parcours s
      std::cout << "use of std" << std::endl;
      tmp=P[i-1]+std::abs(P[i-1]-tmp)*s;
      pasCourant=pas*s;
    }//if
    h++;
    Q[h]=tmp;
    tmp+=pasCourant;
  }//while
}//Interpol

/** Procédure d'interpolation de CPoint par une spline de Catmull-Rom.
 *
 * Comme la classe CVector hérite de la classe CPoint cette procédure
 * peut être utilisée pour interpoler des CVector.
 * 
 *@param n est le nombre de points de contrôle P_i indicés de 0 à n-1 (n >= 4).
 *
 * La courbe passe par les points de contrôle P_1, P_2, ..., P_(n-2).
 *
 * Si P_0=P_1 et P_(n-1)=P_(n-2) alors la courbe passe par tous les points
 * de contrôle. 
 *
 * Il y a n-3 arcs de courbe de degré 3 arcQ_j indicés de 0 à n-4.
 *
 * arcQ_0 est définie par P_0, P_1, P_2, P_3 avec arQ_0(0) = P_1 et arQ_0(1) = P2;
 * arcQ_1 est définie par  P_1, P_2, P_3, P_4 avec arQ_1(0) = P_2 et arQ_1(1) = P3;
 * ....................................
 * Pour j=0..(n-4) arcQ_j est définie par les points P_j, P_(j+1), P_(j+2), P_(j+3)
 * avec arQ_j(0) = P_(j+1) et arcQ_j(1) = P_(j+2)
 *@param nb est le nombre de points souhaités de la courbe d'interpolation.
 * (nb-1) doit être un multiple de n-3.
 *
 *@param P tableau des points de contrôle.
 *@param Q tableau des nb points de la courbe d'interpolation.
 */ 

/*
void CInterpolation::CatmullRom(int n, int nb, std::vector<Point3f> &P, std::vector <Point3f> &Q){
  int k = (nb-1)/(n-3); // k est le nombre de points à calculer sur chaque arcQ_i.
  double t, t2, t3;
  double *tt ;// tableau des valeurs 1/k, 2/k, ..., (k-1)/k
  tt = new double[k];
  tt[0] = 0.0;
  tt[1] = 1.0/(double)k;
  for(int i=2; i<k ; i++){
    tt[i] = tt[1]*i;
  }//for i
  int tQ = 0; // pour parcourir le tableau Q
  for(int j=0; j<n-3; j++){
    // calcul des points de arcQ_j à ranger dans le tableau Q
    Q[tQ] = P[j+1];// arcQ_j débute au point P_(j+1)
    tQ++;
    for(int i=1; i<k; i++){
      // calcul de Q[tQ]=arcQ_j(tt[i])=arcQ_j(i/k)
      t=tt[i];
      t2=t*t;
      t3=t2*t;
      Q[tQ]=P[j]*(0.5*(-t3+2*t2-t))
	+ P[j+1]*(0.5*(3*t3-5*t2+2))
	+ P[j+2]*(0.5*(-3*t3+4*t2+t))
	+ P[j+3]*(0.5*(t3-t2));
      tQ++;
    }//for t
  }//for j
  Q[tQ]=P[n-2]; // le dernier point de la spline est P_(n-2)
  delete [] tt;
}//CatmullRom
void CInterpolation::CatmullRom(int n, int nb, std::vector <Vector3f> &P, std::vector <Vector3f> &Q){
  int k = (nb-1)/(n-3); // k est le nombre de points à calculer sur chaque arcQ_i.
  double t, t2, t3;
  double *tt ;// tableau des valeurs 1/k, 2/k, ..., (k-1)/k
  tt = new double[k];
  tt[0] = 0.0;
  tt[1] = 1.0/(double)k;
  for(int i=2; i<k ; i++){
    tt[i] = tt[1]*i;
  }//for i
  int tQ = 0; // pour parcourir le tableau Q
  for(int j=0; j<n-3; j++){
    // calcul des points de arcQ_j à ranger dans le tableau Q
    Q[tQ] = P[j+1];// arcQ_j débute au point P_(j+1)
    tQ++;
    for(int i=1; i<k; i++){
      // calcul de Q[tQ]=arcQ_j(tt[i])=arcQ_j(i/k)
      t=tt[i];
      t2=t*t;
      t3=t2*t;
      Q[tQ]=P[j]*(0.5*(-t3+2*t2-t))
	+ P[j+1]*(0.5*(3*t3-5*t2+2))
	+ P[j+2]*(0.5*(-3*t3+4*t2+t))
	+ P[j+3]*(0.5*(t3-t2));
      tQ++;
    }//for t
  }//for j
  Q[tQ]=P[n-2]; // le dernier point de la spline est P_(n-2)
  delete [] tt;
}//CatmullRom
*/

/** Procédure d'interpolation d'une suite de double.
 *
 *@param n est le nombre de points de contrôle P_i indicés de 0 à n-1 (n >= 4).
 *
 * La courbe passe par les points de contrôle P_1, P_2, ..., P_(n-2).
 * Si P_0=P_1 et P_(n-1)=P_(n-2) alors la courbe passe par tous les points
 * de contrôle. 
 *
 * Il y a n-3 arcs de courbe de degré 3 arcQ_j indicés de 0 à n-4.
 *
 * arcQ_0 est définie par P_0, P_1, P_2, P_3 avec arQ_0(0) = P_1 et arQ_0(1) = P2;
 * arcQ_1 est définie par  P_1, P_2, P_3, P_4 avec arQ_1(0) = P_2 et arQ_1(1) = P3;
 * ....................................
 * Pour j=0..(n-4) arcQ_j est définie par les points P_j, P_(j+1), P_(j+2), P_(j+3)
 * avec arQ_j(0) = P_(j+1) et arcQ_j(1) = P_(j+2)
 *@param nb est le nombre de points souhaités de la courbe d'interpolation.
 * (nb-1) doit être un multiple de n-3.
 *
 *@param P tableau des points de contrôle.
 *@param Q tableau des nb points de la courbe d'interpolation.
 */

void CInterpolation::CatmullRom(int n, int nb, double * P, double * Q){
  int k = (nb-1)/(n-3); // k est le nombre de points à calculer sur chaque arcQ_i.
  double t, t2, t3, p0, p1, p2, p3;
  double *tt ;// tableau des valeurs 1/k, 2/k, ..., (k-1)/k
  tt = new double[k];
  tt[0] = 0.0;
  tt[1] = 1.0/(double)k;
  for(int i=2; i<k ; i++){
    tt[i] = tt[1]*i;
  }//for i

 
  int tQ=0; // pour parcourir le tableau Q
  for(int j=0; j<n-3; j++){
    // calcul des points de arcQ_j à ranger dans le tableau Q
    p0 = Q[tQ] = P[j+1];// arcQ_j débute au point P_(j+1)
    p1 = 0.5*(P[j+2] - P[j]);
    p2 = P[j] -2.5*P[j+1] + 2*P[j+2] - 0.5*P[j+3];
    p3 = 0.5*(P[j+3] - P[j]) + 1.5*(P[j+1] - P[j+2]);
    tQ++;
    for(int i=1; i<k; i++){
      // calcul de Q[tQ]=arcQ_j(tt[i])=arcQ_j(i/k)
      t=tt[i];
      t2=t*t;
      t3=t2*t;
      Q[tQ] = p0 + t*p1 + t2*p2 + t3*p3;
      tQ++;
    }//for t
  }//for j
  Q[tQ]=P[n-2]; // le dernier point de la spline est P_(n-2)

  delete [] tt;

}// CatmullRom

}//namespace pbrt
