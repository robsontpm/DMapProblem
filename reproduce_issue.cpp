#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <stdexcept>
#include <sstream>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <limits>
#include <stack>
#include <math.h>
#include <cstdio>

// --- Begin include/capd/basicalg/TypeTraits.h ---
//////////////////////////////////////////////////////////////////////////////
//   Package:          CAPD

/////////////////////////////////////////////////////////////////////////////
//
/// @file TypeTraits.h
///
/// @author Tomasz Kapela   @date 2010-03-08
//
/////////////////////////////////////////////////////////////////////////////

// Copyright (C) Tomasz Kapela 2010
//
// This file constitutes a part of the CAPD library,
// distributed under the terms of the GNU General Public License.
// Consult  http://capd.ii.uj.edu.pl/ for details.

#ifndef _CAPD_CAPD_TYPETRAITS_H_
#define _CAPD_CAPD_TYPETRAITS_H_

//#include "capd/interval/Interval.h"
//#include "capd/vectalg/Z2.h"
#include <limits>
#include <algorithm>
#include <cmath>

namespace capd {

/**
 *  Defines type traits such as their values zero, one etc.
 *
 *  CAPD types should define specialization in their header files.
 *
 *  Known specialization are in
 *    - capd/interval/Interval.h
 *    - capd/filib/Interval.h
 *    - capd/interval/IComplex.h
 *    - capd/multiPrec/MpReal.h
 *    - capd/rings/Z2.h
 *    - capd/rings/Zp.h
 *
 */
template <typename T>
class TypeTraits {
public:
  typedef T Real;

  /// returns object set to zero
  static constexpr inline T zero() noexcept{
	return static_cast<T>(0.0);
  }

  /// returns object set to one
  static constexpr inline T one() noexcept{
	return static_cast<T>(1.0);
  }
  static constexpr T max(T a, T b) noexcept;
  static constexpr T min(T a, T b) noexcept;
/* Expected interface
  /// number of decimal digits
  static inline int numberOfDigits() noexcept;

  /// Machine epsilon (the difference between 1 and the least value greater than 1 that is representable).
  static inline T epsilon() noexcept;

  template <typename S>
  static inline T convert(const S & obj){
    return static_cast<T>(obj);
  }
  /// this flag is true for all interval types
  static const bool isInterval = false;

  static constexpr T max(T a, T b) noexcept;
  static constexpr T min(T a, T b) noexcept;
  static constexpr T abs(T a) noexcept;
  static bool isSingular(T a);
  static constexpr bool isInf(T a) noexcept;
  static constexpr bool isNaN(T a) noexcept;
   */

};


/// for given type returns object that represents zero
template <typename T>
inline T zero(){
  return TypeTraits<T>::zero();
}

/// for given type returns object that represents one (identity)
template <typename T>
inline T one(){
  return TypeTraits<T>::one();
}


template <typename T>
class IntegralTypeTraits {
 public:
  typedef T Real;

  /// returns object set to zero
  static inline T zero() {
	return static_cast<T>(0);
  }
  /// returns object set to one
  static inline T one() {
	return static_cast<T>(1);
  }
  /// number of decimal digits
  static inline int numberOfDigits() throw() {
	return std::numeric_limits<T>::digits10;
  }
  /// Machine epsilon (the difference between 1 and the least value greater than 1 that is representable).
  static inline T epsilon() throw() {
	return std::numeric_limits<T>::epsilon();
  }

  template<typename S>
  static inline T convert(const S &obj) {
	return static_cast<T>(obj);
  }
  /// this flag is true for all interval types
  static const bool isInterval = false;

  static constexpr T max(T a, T b) noexcept{
	return std::max(a, b);
  }
  static constexpr T min(T a, T b) noexcept{
	return std::min(a, b);
  }
  static constexpr T abs(T a) noexcept{
	return std::abs(a);
  }

  static bool isSingular(T a){
	return a == TypeTraits<T>::zero();
  }
};

template <typename T>
struct FloatingTypeTraits : public IntegralTypeTraits<T>{

  typedef T Real;

  /// returns object set to zero
  static inline T zero() {
	return static_cast<T>(0.0);
  }
  /// returns object set to one
  static inline T one() {
	return static_cast<T>(1.0);
  }
  /// number of decimal digits
  static inline int numberOfDigits() throw() {
	return std::numeric_limits<T>::digits10;
  }
  /// Machine epsilon (the difference between 1 and the least value greater than 1 that is representable).
  static inline T epsilon() throw() {
	return std::numeric_limits<T>::epsilon();
  }

  using IntegralTypeTraits<T>::convert;
  using IntegralTypeTraits<T>::isInterval;

  static constexpr bool isInf(T a) noexcept{
	return std::isinf(a);
  }
  static constexpr bool isNaN(T a) noexcept{
	return std::isnan(a);
  }
};

/**
 * Traits of type int
 */
template<>
struct TypeTraits<int> : public IntegralTypeTraits<int>{
  using Real = int;
  using T = int;
  static constexpr T max(T a, T b) noexcept{
	return std::max(a, b);
  }
  static constexpr T min(T a, T b) noexcept{
	return std::min(a, b);
  }
};

/**
 * Traits of type short
 */
template<>
struct TypeTraits<short> : public IntegralTypeTraits<short>{
//  using IntegralTypeTraits<short>::Real;
//  using IntegralTypeTraits<short>::convert;
};

/**
 * Traits of type long
 */
template<>
struct TypeTraits<long> : public IntegralTypeTraits<long>{
//  using IntegralTypeTraits<long>::Real;
//  using IntegralTypeTraits<long>::convert;
};

/**
 * Traits of type long long
 */
template<>
struct TypeTraits<long long> : public IntegralTypeTraits<long long>{
//  using IntegralTypeTraits<long long>::Real;
//  using IntegralTypeTraits<long long>::convert;
};

/**
 * Traits of type double
 */
template<>
struct TypeTraits<double> : public FloatingTypeTraits<double>{
  //using FloatingTypeTraits<double>::Real;
//  using FloatingTypeTraits<double>::convert;
};

/**
 * Traits of type float
 */
template<>
struct TypeTraits<float> : public FloatingTypeTraits<float>{
//  using FloatingTypeTraits<float>::Real;
//  using FloatingTypeTraits<float>::convert;
};

/**
 * Traits of type long double
 */
template<>
struct TypeTraits<long double> : public FloatingTypeTraits<long double>{
//  using FloatingTypeTraits<long double>::Real;
//  using FloatingTypeTraits<long double>::convert;
};


template <typename T>
class TypeTraits<T*> {
public:

  /// returns object set to zero
  static inline T* zero(){
    return static_cast<T*>(0);
  }
};

} // end of namespace capd

#endif // _CAPD_CAPD_TYPETRAITS_H_

// --- End include/capd/basicalg/TypeTraits.h ---

// --- Begin include/capd/basicalg/minmax.h ---


/////////////////////////////////////////////////////////////////////////////
/// @file minmax.h
///
/// @author The CAPD Group
/////////////////////////////////////////////////////////////////////////////

// Copyright (C) 2000-2005 by the CAPD Group.
//
// This file constitutes a part of the CAPD library,
// distributed under the terms of the GNU General Public License.
// Consult  http://capd.ii.edu.pl/ for details.

/* min, max and abs definitions */

#ifndef _CAPD_BASICALG_MINMAX_H_
#define _CAPD_BASICALG_MINMAX_H_

// #include "capd/basicalg/TypeTraits.h"

#undef max
#undef min

namespace capd {
/// @addtogroup capd
/// @{

//
// The following lines was prone to errors
//
//template<typename T>
//inline T min(const T& x, const T& y) {
//  return (x<y ? x : y);
//}
//
//template<typename T>
//inline T max(const T& x, const T& y) {
//  return (x<y ? y : x);
//}


  template<typename T>
  inline T min(const T &x, const T &y) {
	return ::capd::TypeTraits<T>::min(x, y);
  }

  template<typename T>
  inline T max(const T &x, const T &y) {
	return ::capd::TypeTraits<T>::max(x, y);
  }

  template<typename T>
  inline T abs(const T &x) {
	return ::capd::TypeTraits<T>::abs(x);
  }
//
//  inline long double abs(long double x) {
//	return (x < 0.) ? -x : x;
//  }
//
//  inline double abs(double x) {
//	return (x < 0.) ? -x : x;
//  }
//
//  inline int abs(int x) {
//	return (x < 0.) ? -x : x;
//  }

/// @}
}
#endif // _CAPD_BASICALG_MINMAX_H_


// --- End include/capd/basicalg/minmax.h ---

// --- Begin include/capd/basicalg/power.h ---
/////////////////////////////////////////////////////////////////////////////
/// @file power.h
///
/// @author The CAPD Group
/////////////////////////////////////////////////////////////////////////////

// Copyright (C) 2000-2005 by the CAPD Group.
//
// This file constitutes a part of the CAPD library,
// distributed under the terms of the GNU General Public License.
// Consult  http://capd.wsb-nlu.edu.pl/ for details.

#ifndef _CAPD_CAPD_POWER_H_
#define _CAPD_CAPD_POWER_H_

#include <cmath>
#include <math.h> // required by the Borland compiler

/// @addtogroup basicalg
/// @{

inline double power(double value, int exponent)
{
   return ::pow(value,exponent);
}

inline long double power(long double value, int exponent)
{
   return ::powl(value,exponent);
}

inline float power(float value, int exponent)
{
   return ::pow(value,exponent);
}

inline double power(int value, int exponent)
{
   return ::pow(static_cast<double> (value),exponent);
}

inline double sqr(double x)
{
  return x*x;
}

inline float sqr(float x)
{
  return x*x;
}

inline long double sqr(long double x)
{
  return x*x;
}

/// @}

#endif // _CAPD_CAPD_POWER_H_

// --- End include/capd/basicalg/power.h ---

// --- Begin include/capd/basicalg/factrial.h ---

/////////////////////////////////////////////////////////////////////////////
/// @file factrial.h
///
/// @author The CAPD Group
/////////////////////////////////////////////////////////////////////////////

// Copyright (C) 2000-2005 by the CAPD Group.
//
// This file constitutes a part of the CAPD library,
// distributed under the terms of the GNU General Public License.
// Consult  http://capd.ii.uj.edu.pl/ for details.
#include <vector>
#ifndef _CAPD_CAPD_FACTRIAL_H_
#define _CAPD_CAPD_FACTRIAL_H_
#include <stdexcept>
namespace capd{

class Newton{
public:
  static Newton& getInstance(){
    return instance;
  }
  unsigned long long factorial(unsigned n);
//  unsigned long long newton(unsigned n, unsigned k);
  unsigned long long newton(unsigned n, unsigned k)
{
  unsigned first_undefined_index=index(first_unknown_newton_level,0);
	if(n>=first_unknown_newton_level){
      newton_storage.resize(index(n+1,0));
		if(first_undefined_index == 0){
			newton_storage[first_undefined_index++]=1;
			first_unknown_newton_level++;
		}
		for(unsigned m=first_unknown_newton_level;m<=n;m++){
			newton_storage[first_undefined_index]=newton_storage[first_undefined_index+m]=1;
			for(unsigned p=1;p<m;p++) newton_storage[first_undefined_index+p]=
					newton_storage[index(m-1,p-1)]+newton_storage[index(m-1,p)];
			first_undefined_index+=(m+1);
		}
		first_unknown_newton_level=n+1;
      }
	return newton_storage[index(n,k)];
}

private:
  Newton() : first_unknown_factorial(0), first_unknown_newton_level(0) {}
  std::vector<unsigned long long> factorial_storage;
  unsigned first_unknown_factorial;
  std::vector<unsigned long long> newton_storage;
  unsigned first_unknown_newton_level;

  inline unsigned index(unsigned n,unsigned k)
  {
    return n*(n+1)/2+k;
  }
  static Newton instance;
};

/// @addtogroup basicalg
/// @{
template <long N, long K>
struct Binomial
{
   static const long value = Binomial<N-1,K-1>::value + Binomial<N-1,K>::value;
};

template <long K>
struct Binomial<0,K>
{
   static const long value = 0;
};

template <long N>
struct Binomial<N,0>
{
   static const long value = 1;
};

template<long N>
struct Binomial<N,N>
{
   static const long value=1;
};

template <>
struct Binomial<0,0>
{
   static const long value = 1;
};

template<long N>
struct Factorial
{
   static const long value = N*Factorial<N-1>::value;
};

template<>
struct Factorial<1>
{
   static const long value = 1;
};

template<>
struct Factorial<0>
{
   static const long value = 1;
};

inline double realFactorial(unsigned n){
  static double factorials[] = {
    1.0,
    1.0,
    2.0,
    6.0,
    24.0,
    120.0,
    720.0,
    5040.0,
    40320.0,
    362880.0,
    3628800.0,
    39916800.0,
    479001600.0,
    6227020800.0,
    87178291200.0,
    1307674368000.0,
    20922789888000.0,
    355687428096000.0,
    6402373705728000.0,
    121645100408832000.0,
    2432902008176640000.0
  };
  if(n<=20)
    return factorials[n];
  else
    throw std::runtime_error("Function realFactorial: n! exceeded double capacity.");
}
/// @}

} // namespace capd

///< compute and store n factorial
inline unsigned long long factorial(unsigned n){
  return capd::Newton::getInstance().factorial(n);
}


///< compute and store newton symbol (n \over k)
inline unsigned long long binomial(unsigned n, unsigned k){
  return capd::Newton::getInstance().newton(n,k);
}

#endif // _CAPD_CAPD_FACTRIAL_H_

// --- End include/capd/basicalg/factrial.h ---

// --- Begin factrial.cpp ---
/// @addtogroup capd
/// @{

/////////////////////////////////////////////////////////////////////////////
/// @file factrial.cpp
///
/// @author The CAPD Group
/////////////////////////////////////////////////////////////////////////////

// Copyright (C) 2000-2013 by the CAPD Group.
//
// This file constitutes a part of the CAPD library,
// distributed under the terms of the GNU General Public License.
// Consult  http://capd.ii.uj.edu.pl/ for details.

#include <cstddef>
// #include "capd/basicalg/factrial.h"

using namespace capd;

unsigned long long Newton::factorial(unsigned n)
{
	if(n<first_unknown_factorial){
		return factorial_storage[n];
	}else{
	  unsigned i;
    factorial_storage.resize(n+1);
		if(first_unknown_factorial == 0){
			factorial_storage[first_unknown_factorial++]=1;
		}
		unsigned long long result=factorial_storage[first_unknown_factorial-1];
		for(i=first_unknown_factorial;i<=n;i++) factorial_storage[i]=result*=i;
		first_unknown_factorial=n+1;
		return result;
	}
}
/*
unsigned long long Newton::newton(unsigned n, unsigned k)
{
  unsigned first_undefined_index=index(first_unknown_newton_level,0);
	if(n>=first_unknown_newton_level){
      newton_storage.resize(index(n+1,0));
		if(first_undefined_index == 0){
			newton_storage[first_undefined_index++]=1;
			first_unknown_newton_level++;
		}
		for(unsigned m=first_unknown_newton_level;m<=n;m++){
			newton_storage[first_undefined_index]=newton_storage[first_undefined_index+m]=1;
			for(unsigned p=1;p<m;p++) newton_storage[first_undefined_index+p]=
					newton_storage[index(m-1,p-1)]+newton_storage[index(m-1,p)];
			first_undefined_index+=(m+1);
		}
		first_unknown_newton_level=n+1;
      }
	return newton_storage[index(n,k)];
}
*/
Newton Newton::instance;
/// @}

// --- End factrial.cpp ---

// --- Begin include/capd/vectalg/algebraicOperations.h ---
/// @addtogroup vectalg
/// @{

/////////////////////////////////////////////////////////////////////////////
/// @file algebraicOperations.h
///
/// This file provides an algebraic operation which may be implemented on the container level
/// such as addition of some objects
///
/// Constraints on any type which appears in these algorithms:
/// - public typedef ScalarType
/// - public types const_iterator and iterator and corresponding functions begin(), end()
/// - public const function dimension() which returns an object which can be used to initialize other objects
///
/// @author Daniel Wilczak
/////////////////////////////////////////////////////////////////////////////

// Copyright (C) 2000-2005 by the CAPD Group.
//
// This file constitutes a part of the CAPD library,
// distributed under the terms of the GNU General Public License.
// Consult  http://capd.ii.uj.edu.pl/ for details.




#ifndef _CAPD_VECTALG_ALGEBRAICOPERATIONS_H_
#define _CAPD_VECTALG_ALGEBRAICOPERATIONS_H_

#include <stdexcept>
// #include "capd/basicalg/minmax.h"
#include <sstream>

inline double nonnegativePart(double x){
  return capd::abs(x);
}

inline double nonnegativePart(long double x){
  return capd::abs(x);
}

inline double nonnegativePart(int x){
  return capd::abs(x);
}



namespace capd{
namespace vectalg{

/// Assign zero to each coordinate
template<typename Object>
void clear(Object& u);

/// Computes euclidean norm of any vector
template<typename Object>
typename Object::ScalarType euclNorm(const Object& u);

/// normalize a vector with respect to euclidean norm
/// if impossible returns false
template<typename Object>
bool normalize(Object& u);

//------------------ unary arithmetic operations ------------------//

template<typename ResultType, typename Object>
ResultType absoluteValue(const Object& v);


template<typename ResultType, typename Object>
ResultType unaryMinus(const Object& v);

//-----------------------arithmetic operations----------------------//
/**
  this procedure can be use to add two vector-like objects:
  - vectors, matrices, references to columns or rows of matrix
  - higher dimensional containers
  result = v1 + v2
*/
template<typename ResultType, typename T1, typename T2>
void addObjects(const T1& v1,const T2& v2, ResultType& result);

template<typename ResultType, typename T1, typename T2>
inline
ResultType addObjects(const T1& v1,const T2& v2)
{
  if(v1.dimension()!=v2.dimension())
    throw std::range_error("capd::vectalg::addObjects: Incompatible dimensions");
  ResultType result(v1.dimension(),true);
  addObjects(v1,v2,result);
  return result;
}


/**
  this procedure can be use to subtract two vector-like objects:
  - vectors, matrices, references to columns or rows of matrix
  - higher dimensional containers
  result = v1 - v2
*/
template<typename ResultType, typename T1, typename T2>
void subtractObjects(const T1& v1,const T2& v2, ResultType& result);

template<typename ResultType, typename T1, typename T2>
inline
ResultType subtractObjects(const T1& v1,const T2& v2)
{
  if(v1.dimension()!=v2.dimension())
    throw std::range_error("capd::vectalg::addObjects: Incompatible dimensions");
  ResultType result(v1.dimension(),true);
  subtractObjects(v1,v2,result);
  return result;
}

/**
  this procedure can be use to compute scalar product of two vector-like objects:
  - vectors, matrices, references to columns or rows of matrix
  - vectors of partial derivatives from higher order containers
  result = v1 * v2
*/
template<typename T1, typename T2>
typename T1::ScalarType scalarProduct(const T1& v1,const T2& v2);

/**
  this procedure can be use to multiply by a scalar any element of vector-like objects
  as a result we may obtain object of different type,
    multiplication of column of matrix and scalar gives vector
  result = v * s
*/
template<typename ResultType, typename Object, typename FactorType>
ResultType multiplyObjectScalar(const Object& v,const FactorType& s);

/**
  this procedure can be use to divide by a scalar any element of vector-like objects
  as a result we may obtain object of different type,
    dividing of column of matrix by scalar gives vector
  result = v / s
*/
template<typename ResultType, typename Object, typename FactorType>
ResultType divideObjectScalar(const Object& v,const FactorType& s);

/**
  this procedure can be use to add a scalar to any element of vector-like objects
  result[i] = v[i] + s
*/
template<typename ResultType, typename Object, typename FactorType>
ResultType addObjectScalar(const Object& v,const FactorType& s);

/**
  this procedure can be used to substract a scalar from any element of vector-like objects
  result[i] = v[i] - s
*/
template<typename ResultType, typename Object, typename FactorType>
ResultType subtractObjectScalar(const Object& v,const FactorType& s);

/**
  this procedure realizes multiplication of matrix-like object by vector-like object
  result = m*v
*/
template<typename ResultType, typename MatrixType, typename VectorType>
void matrixByVector(const MatrixType& m,const VectorType& u, ResultType& result);

template<typename ResultType, typename MatrixType, typename VectorType>
inline
ResultType matrixByVector(const MatrixType& m,const VectorType& u)
{
  ResultType result(m.numberOfRows(),true);
  if(m.numberOfColumns()!=u.dimension())
    throw std::range_error("operator Matrix*Vector: incompatible dimensions");
  matrixByVector(m,u,result);
  return result;
}

/**
  this procedure realizes multiplication of two matrix-like objects
  result = m1*m2
*/
template<typename ResultType, typename Matrix1, typename Matrix2>
void matrixByMatrix(const Matrix1& a1, const Matrix2& a2, ResultType& result);

template<typename ResultType, typename Matrix1, typename Matrix2>
inline
ResultType matrixByMatrix(const Matrix1& a1, const Matrix2& a2)
{
  ResultType result(a1.numberOfRows(),a2.numberOfColumns(),true);
  if(a1.numberOfColumns()!=a2.numberOfRows()) {
    std::stringstream msg;
    msg << "operator capd::vectalg::matrixByMatrix: incompatible matrix dimensions ";
    msg << "[" << a1.numberOfRows() << ", " << a1.numberOfColumns() << "] * ";
    msg << "[" << a2.numberOfRows() << ", " << a2.numberOfColumns() << "]";
    throw std::range_error(msg.str());
  }
  matrixByMatrix(a1,a2,result);
  return result;
}

//----------------------assignments - objects---------------------------//

/**
  this procedure can be use to assign one vector-like objects
  from the other.
  u = v
*/
template<typename T1, typename T2>
T1& assignObjectObject(T1& u, const T2& v);

/**
  this procedure can be use to add of two vector-like objects
  result is stored in the first argument
  u += v
*/
template<typename T1, typename T2>
T1& addAssignObjectObject(T1& u, const T2& v);

/**
  this procedure can be use to subtract of two vector-like objects
  result is stored in the first argument
  u -= v
*/
template<typename T1, typename T2>
T1& subtractAssignObjectObject(T1& u, const T2& v);

//----------------------assignments - Scalars---------------------------//

/**
  this procedure can be use to assign each element of a vector-like object
  to be equal to a given scalar
  u[i] = s
*/
template<typename Object,typename Scalar>
Object& assignFromScalar(Object& u, const Scalar& s);

/**
  this procedure can be use to add a scalar to each element of a vector-like object
  u[i] += s
*/
template<typename Object,typename Scalar>
Object& addAssignObjectScalar(Object& u, const Scalar& s);

/**
  this procedure can be use to subtract a scalar from each element of a vector-like object
  u[i] -= s
*/
template<typename Object,typename Scalar>
Object& subtractAssignObjectScalar(Object& u, const Scalar& s);

/**
  this procedure can be use to multiply by a scalar each element of a vector-like object
  u[i] *= s
*/
template<typename Object,typename Scalar>
Object& multiplyAssignObjectScalar(Object& u, const Scalar& s);

/**
  this procedure can be use to multiply by a scalar each element of a vector-like object
  and then add compoent-wise elements of second vector-like object
  u[i] = u[i]*s+v[i]
*/
template<typename Object, typename Object2, typename Scalar>
Object& multiplyAssignObjectScalarAddObject(Object& u, const Scalar& s, const Object2& v);

/**
  This procedure computes
  u += A*v
  where u,v are vectors and A is a matrix
*/
template<typename V, typename M, typename V2>
V& addAssignMatrixByVector(V& u, const M& A, const V2& v);

/**
  This procedure computes
  u -= A*v
  where u,v are vectors and A is a matrix
*/
template<typename V, typename M, typename V2>
V& subtractAssignMatrixByVector(V& u, const M& A, const V2& v);

/**
  this procedure can be use to divide by a scalar any element of two vector-like objects
  u[i] /= s
*/
template<typename Object,typename Scalar>
Object& divideAssignObjectScalar(Object& u, const Scalar& s);

/**
 * Evaluates polynomial p:R->R^n at some argument.
 * Coefficients of the polynomial are assumed to be stored as an array c[r],
 * where c\in R^n. The result is stored in the third argument.
 *
 * @param c - coefficients of the polynomial
 * @param t - argument at which polynomial is evaluated
 * @param n - deree of the polynomial
 */
template<class Object, class Scalar>
void evalPolynomial(Object* c, Scalar t, Object& result, int n);

//-------coord-wise inequalities - true if true for each coord---------//

template<typename T1,typename T2>
bool lessThan (const T1& v1, const T2& v2);

template<typename T1,typename T2>
bool greaterThan (const T1& v1, const T2& v2);

template<typename T1,typename T2>
bool lessEqual (const T1& v1, const T2& v2);

template<typename T1,typename T2>
bool greaterEqual (const T1& v1, const T2& v2);

// ---------------------------------------- equality and not equality --------------------

template<typename T1,typename T2>
bool equal (const T1& v1, const T2& v2);

template<typename T1,typename T2>
bool notEqual (const T1& v1, const T2& v2)
{
  return ! equal(v1,v2);
}


}} // end of namespace capd::vectalg

#endif // _CAPD_VECTALG_ALGEBRAICOPERATIONS_H_

/// @}

// --- End include/capd/vectalg/algebraicOperations.h ---

// --- Begin include/capd/vectalg/Container.h ---
/// @addtogroup vectalg
/// @{

/////////////////////////////////////////////////////////////////////////////
/// @file Container.h
///
/// This file provides a template class Container together with suitable iterators
/// The container has fixed size specified by a template argument 'capacity'
///
/// Also a specialization of this class for capacity=0 is defined
/// In that case objects in this container are allocated on storage instead of stack
///
/// This class is used as a container for vectors, matrices and higher order containers
///
/// @author Daniel Wilczak
/////////////////////////////////////////////////////////////////////////////

// Copyright (C) 2000-2005 by the CAPD Group.
//
// This file constitutes a part of the CAPD library,
// distributed under the terms of the GNU General Public License.
// Consult  http://capd.ii.uj.edu.pl/ for details.


#ifndef _CAPD_VECTALG_CONTAINER_H_
#define _CAPD_VECTALG_CONTAINER_H_

#include <stdexcept>
#include <cstdlib> // for size_t
//#include "capd/settings/compilerSetting.h"
//#include "capd/auxil/Dll.h"

namespace capd{
namespace vectalg{

typedef unsigned __size_type;
typedef int __difference_type; // must be signed integral type

/// class Container together with suitable iterators
/// The container has fixed size specified by a template argument 'capacity'
///
/// This class is used as a container for vectors, matrices and higher order containers
template<typename Scalar, __size_type capacity>
class Container
{
public:
  typedef Scalar ScalarType;
  typedef __size_type size_type;
  typedef __difference_type difference_type;
  typedef ScalarType* iterator;
  typedef const ScalarType* const_iterator;
  typedef std::reverse_iterator<iterator> reverse_iterator;
  typedef std::reverse_iterator<const_iterator> const_reverse_iterator;

  Container();
  explicit Container(size_type);
  Container(size_type,bool); // it does not insert zeros

  iterator begin();
  iterator end();
  const_iterator begin() const;
  const_iterator end() const;

  reverse_iterator rbegin();
  reverse_iterator rend();
  const_reverse_iterator rbegin() const;
  const_reverse_iterator rend() const;

  /* Container& operator=(const Container&);*/
  void resize(size_type newCapacity);

  ScalarType& operator[](size_type);
  const ScalarType& operator[](size_type) const;
  ScalarType& operator()(size_type);
  const ScalarType& operator()(size_type) const;
  friend void swap(Container<Scalar,capacity>& A_c1, Container<Scalar,capacity>& A_c2)
  {
    iterator b=A_c1.begin(), e=A_c1.end();
    iterator i = A_c2.begin();
    while(b!=e)
    {
      std::swap(*b,*i);
      ++b;
      ++i;
    }
  }
  void clear();
// memory allocation
  static size_type size() {return capacity;}

protected:
  ScalarType data[capacity];
};

/// Specialization for capacity=0
/// This container allocates objects on a storage
template<typename Scalar>
class Container<Scalar,0>
{
public:
  typedef Scalar ScalarType;
  typedef __size_type size_type;
  typedef __difference_type difference_type;
  typedef ScalarType* iterator;
  typedef const ScalarType* const_iterator;
  typedef std::reverse_iterator<iterator> reverse_iterator;
  typedef std::reverse_iterator<const_iterator> const_reverse_iterator;

  Container& operator=(Container&&) noexcept;
  Container(Container&&) noexcept;

  Container();
  explicit Container(size_type);
  Container(size_type,bool); // it does not insert zeros
  Container(const Container&);
  ~Container();

  iterator begin();
  iterator end();
  const_iterator begin() const;
  const_iterator end() const;

  reverse_iterator rbegin();
  reverse_iterator rend();
  const_reverse_iterator rbegin() const;
  const_reverse_iterator rend() const;

  Container& operator=(const Container&);
  void resize(size_type);

  ScalarType& operator[](size_type);
  const ScalarType& operator[](size_type) const;
  ScalarType& operator()(size_type);
  const ScalarType& operator()(size_type) const;
  friend void swap(Container<Scalar,0>& A_c1, Container<Scalar,0>& A_c2)
  {
     std::swap(A_c1.data,A_c2.data);
     std::swap(A_c1.capacity,A_c2.capacity);
  }

  size_type size() const {return capacity;}
  bool empty() const { return capacity == 0; }

  void clear();

protected:
  ScalarType *data;
  size_type capacity;
};

// ---- inline definitions for Containers ----------------- //

template<typename Scalar>
inline Container<Scalar,0>::Container() : data(0), capacity(0)
{
}
// --------------- iterator selection --------------------- //

template<typename Scalar, __size_type capacity>
inline typename Container<Scalar,capacity>::iterator Container<Scalar,capacity>::begin()
{
  return iterator(data);
}

template<typename Scalar, __size_type capacity>
inline typename Container<Scalar,capacity>::iterator Container<Scalar,capacity>::end()
{
  return iterator(data+capacity);
}

template<typename Scalar, __size_type capacity>
inline typename Container<Scalar,capacity>::const_iterator Container<Scalar,capacity>::begin() const
{
  return const_iterator(data);
}

template<typename Scalar, __size_type capacity>
inline typename Container<Scalar,capacity>::const_iterator Container<Scalar,capacity>::end() const
{
  return const_iterator(data+capacity);
}

template<typename Scalar, __size_type capacity>
inline typename Container<Scalar,capacity>::reverse_iterator Container<Scalar,capacity>::rbegin()
{
  return reverse_iterator(end());
}

template<typename Scalar, __size_type capacity>
inline typename Container<Scalar,capacity>::reverse_iterator Container<Scalar,capacity>::rend()
{
  return reverse_iterator(begin());
}

template<typename Scalar, __size_type capacity>
inline typename Container<Scalar,capacity>::const_reverse_iterator Container<Scalar,capacity>::rbegin() const
{
  return const_reverse_iterator(end());
}

template<typename Scalar, __size_type capacity>
inline typename Container<Scalar,capacity>::const_reverse_iterator Container<Scalar,capacity>::rend() const
{
  return const_reverse_iterator(begin());
}

template<typename Scalar>
inline typename Container<Scalar,0>::iterator Container<Scalar,0>::begin()
{
  return iterator(data);
}

template<typename Scalar>
inline typename Container<Scalar,0>::iterator Container<Scalar,0>::end()
{
  return iterator(data+capacity);
}

template<typename Scalar>
inline typename Container<Scalar,0>::const_iterator Container<Scalar,0>::begin() const
{
  return const_iterator(data);
}

template<typename Scalar>
inline typename Container<Scalar,0>::const_iterator Container<Scalar,0>::end() const
{
  return const_iterator(data+capacity);
}

template<typename Scalar>
inline typename Container<Scalar, 0>::reverse_iterator Container<Scalar, 0>::rbegin()
{
  return reverse_iterator(end());
}

template<typename Scalar>
inline typename Container<Scalar, 0>::reverse_iterator Container<Scalar, 0>::rend()
{
  return reverse_iterator(begin());
}

template<typename Scalar>
inline typename Container<Scalar, 0>::const_reverse_iterator Container<Scalar, 0>::rbegin() const
{
  return const_reverse_iterator(end());
}

template<typename Scalar>
inline typename Container<Scalar, 0>::const_reverse_iterator Container<Scalar, 0>::rend() const
{
  return const_reverse_iterator(begin());
}

// ------------------------- indexing ------------------------ //

template<typename Scalar, __size_type capacity>
inline Scalar& Container<Scalar,capacity>::operator[] (size_type i)
{
  return data[i];
}

template<typename Scalar, __size_type capacity>
inline const Scalar& Container<Scalar,capacity>::operator[] (size_type i) const
{
  return data[i];
}

template<typename Scalar, __size_type capacity>
inline Scalar& Container<Scalar,capacity>::operator() (size_type i)
{
  return data[i-1];
}

template<typename Scalar, __size_type capacity>
inline const Scalar& Container<Scalar,capacity>::operator() (size_type i) const
{
  return data[i-1];
}

template<typename Scalar>
inline Scalar& Container<Scalar,0>::operator[] (size_type i)
{
  return data[i];
}

template<typename Scalar>
inline const Scalar& Container<Scalar,0>::operator[] (size_type i) const
{
  return data[i];
}

template<typename Scalar>
inline Scalar& Container<Scalar,0>::operator() (size_type i)
{
  return data[i-1];
}

template<typename Scalar>
inline const Scalar& Container<Scalar,0>::operator() (size_type i) const
{
  return data[i-1];
}

// ------------ constructor - desctructor --------------------

template<typename Scalar, __size_type capacity>
inline Container<Scalar,capacity>::Container(size_type,bool)
{}

template<typename Scalar>
inline Container<Scalar,0>::Container(size_type a_capacity,bool) : capacity(a_capacity)
{
  data = new ScalarType[capacity];
}

template<typename Scalar>
inline Container<Scalar,0>::~Container()
{
  if(data) delete [] data;
}

}} // namespace capd::vectalg

#endif // _CAPD_VECTALG_CONTAINER_H_

/// @}

// --- End include/capd/vectalg/Container.h ---

// --- Begin include/capd/vectalg/Container.hpp ---
/// @addtogroup vectalg
/// @{

/////////////////////////////////////////////////////////////////////////////
/// @file Container.hpp
///
/// @author Daniel Wilczak 2005-2008
/////////////////////////////////////////////////////////////////////////////

// Copyright (C) 2000-2005 by the CAPD Group.
//
// This file constitutes a part of the CAPD library,
// distributed under the terms of the GNU General Public License.
// Consult  http://capd.ii.uj.edu.pl/ for details.

#ifndef _CAPD_VECTALG_CONTAINER_HPP_
#define _CAPD_VECTALG_CONTAINER_HPP_

#include <algorithm>
// #include "capd/vectalg/Container.h"
// #include "capd/basicalg/TypeTraits.h"

#include <iostream>
namespace capd{
namespace vectalg{

  /// Move constructor
template<typename Scalar>
Container<Scalar,0>::Container(Container && a_container) noexcept
  : data(a_container.data), capacity(a_container.capacity) {
  a_container.data = 0;
  a_container.capacity = 0;
}

template<typename Scalar>
Container<Scalar,0>& Container<Scalar,0>::operator=(Container&& a_c) noexcept{
//  Self assigment check removed: it is rare.
//  if(&a_c != this) {
    std::swap(capacity, a_c.capacity);
    std::swap(data, a_c.data);
//  }
  return *this;
}

// --------------- member definitions ----------------------------- //

template<typename Scalar, __size_type capacity>
void Container<Scalar,capacity>::clear(){
  std::fill(begin(), end(), TypeTraits<ScalarType>::zero());
}

template<typename Scalar>
void Container<Scalar,0>::clear(){
  std::fill(begin(), end(), TypeTraits<ScalarType>::zero());
}

template<typename Scalar, __size_type capacity>
Container<Scalar,capacity>::Container(){
   clear();
}

template<typename Scalar, __size_type capacity>
Container<Scalar,capacity>::Container(size_type){
  clear();
}

/*
template<typename Scalar, __size_type capacity>
Container<Scalar,capacity>& Container<Scalar,capacity>::operator=(const Container& a_c)
{
  if(&a_c != this)
    std::copy(a_c.begin(),a_c.end(),begin());
  return *this;
}
*/

template<typename Scalar, __size_type capacity>
void Container<Scalar,capacity>::resize(size_type newCapacity)
{
  if(newCapacity!=capacity)
    throw std::range_error("Cannot change capacity of static container");
}

template<typename Scalar>
Container<Scalar,0>::Container(size_type a_capacity) : capacity(a_capacity)
{
  data = new ScalarType[capacity];
  clear();
}


template<typename Scalar>
Container<Scalar,0>::Container(const Container& a_container) : capacity(a_container.capacity)
{
  data = new ScalarType[capacity];
  std::copy(a_container.begin(),a_container.end(),begin());
}

template<typename Scalar>
Container<Scalar,0>& Container<Scalar,0>::operator=(const Container& a_c)
{
  if(&a_c != this)
  {
    if(capacity!=a_c.capacity)
    {
      delete [] data;
      capacity =  a_c.capacity;
      data = new ScalarType[capacity];
    }
    std::copy(a_c.begin(),a_c.end(),begin());
  }
  return *this;
}

template<typename Scalar>
void Container<Scalar,0>::resize(size_type A_newCapacity)
{
  if(capacity!=A_newCapacity)
  {
    if(data) delete[] data;
    capacity = A_newCapacity;
    data = new ScalarType[capacity];
  }
  clear();
}

}} // namespace capd::vectalg

#endif // _CAPD_VECTALG_CONTAINER_HPP_

/// @}

// --- End include/capd/vectalg/Container.hpp ---

// --- Begin include/capd/vectalg/algebraicOperations.hpp ---
/// @addtogroup vectalg
/// @{

/////////////////////////////////////////////////////////////////////////////
/// @file algebraicOperations.hpp
///
/// This file provides an algebraic operation which may be implemented on the container level
/// such as addition of some objects
///
/// Constraints on any type which appears in these algorithms:
/// - public typedef ScalarType
/// - public types const_iterator and iterator and corresponding functions begin(), end()
/// - public const function dimension() which returns an object which can be used to initialize other objects
///
/// @author Daniel Wilczak
/////////////////////////////////////////////////////////////////////////////

// Copyright (C) 2000-2013 by the CAPD Group.
//
// This file constitutes a part of the CAPD library,
// distributed under the terms of the GNU General Public License.
// Consult  http://capd.ii.uj.edu.pl/ for details.




#ifndef _CAPD_VECTALG_ALGEBRAICOPERATIONS_HPP_
#define _CAPD_VECTALG_ALGEBRAICOPERATIONS_HPP_

#include <stdexcept>
// #include "capd/basicalg/minmax.h"
// #include "capd/vectalg/algebraicOperations.h"
// #include "capd/basicalg/TypeTraits.h"

namespace capd{
namespace vectalg{


/// Assign zero to each coordinate
template<typename Object>
void clear(Object& u){
  typename Object::iterator b=u.begin(), e=u.end();
  while(b!=e)
  {
    *b = TypeTraits<typename Object::ScalarType>::zero();
    ++b;
  }
}

/// Computes euclidean norm of any vector
template<typename Object>
typename Object::ScalarType euclNorm(const Object& u){
  typedef typename Object::ScalarType Scalar;
  Scalar sum = TypeTraits<Scalar>::zero();
  typename Object::const_iterator b=u.begin(), e=u.end();
  while(b!=e)
  {
    sum += static_cast<Scalar>(power(*b,2));
    ++b;
  }
  return Scalar(sqrt(nonnegativePart(sum)));
}

/// normalize a vector with respect to euclidean norm
/// if impossible returns false
template<typename Object>
bool normalize(Object& u){
  typedef typename Object::ScalarType Scalar;
  Scalar n;
  if( ! (( n=euclNorm(u)) > 0.0) )
  {
    return false;
  }else{
    typename Object::iterator b=u.begin(), e=u.end();
    while(b!=e)
    {
      *b /=n;
      ++b;
    }
    return true;
  }
}

//------------------ unary arithmetic operations ------------------//

template<typename ResultType, typename Object>
ResultType absoluteValue(const Object& v)
{
  ResultType result(v.dimension(),true);
  typename ResultType::iterator b=result.begin(), e=result.end();
  typename Object::const_iterator i=v.begin();
  while(b!=e)
  {
    *b = capd::abs(*i);
    ++b;
    ++i;
  }
  return result;
}


template<typename ResultType, typename Object>
ResultType unaryMinus(const Object& v)
{
  ResultType result(v.dimension(),true);
  typename ResultType::iterator b=result.begin(), e=result.end();
  typename Object::const_iterator i=v.begin();
  while(b!=e)
  {
    *b = -(*i);
    ++b;
    ++i;
  }
  return result;
}

//-----------------------arithmetic operations----------------------//
/**
  this procedure can be use to add two vector-like objects:
  - vectors, matrices, references to columns or rows of matrix
  - higher dimensional containers
  result = v1 + v2
*/
template<typename ResultType, typename T1, typename T2>
void addObjects(const T1& v1,const T2& v2, ResultType& result)
{
  typename ResultType::iterator b=result.begin(), e=result.end();
  typename T1::const_iterator b1=v1.begin();
  typename T2::const_iterator b2=v2.begin();
  while(b!=e)
  {
    *b = (*b1) + (*b2);
    ++b;
    ++b1;
    ++b2;
  }
}


/**
  this procedure can be use to subtract two vector-like objects:
  - vectors, matrices, references to columns or rows of matrix
  - higher dimensional containers
  result = v1 - v2
*/
template<typename ResultType, typename T1, typename T2>
void subtractObjects(const T1& v1,const T2& v2, ResultType& result)
{
  typename ResultType::iterator b=result.begin(), e=result.end();
  typename T1::const_iterator b1=v1.begin();
  typename T2::const_iterator b2=v2.begin();
  while(b!=e)
  {
    *b = (*b1) - (*b2);
    ++b;
    ++b1;
    ++b2;
  }
}

/**
  this procedure can be use to compute scalar product of two vector-like objects:
  - vectors, matrices, references to columns or rows of matrix
  - vectors of partial derivatives from higher order containers
  result = v1 * v2
*/
template<typename T1, typename T2>
typename T1::ScalarType scalarProduct(const T1& v1,const T2& v2)
{
  if(v1.dimension()!=v2.dimension())
    throw std::range_error("capd::vectalg::scalarProduct: Incompatible dimensions");
  typename T1::ScalarType result(0);
  typename T1::const_iterator b1=v1.begin();
  typename T2::const_iterator b2=v2.begin(), e2=v2.end();
  while(b2!=e2)
  {
    result += (*b1) * (*b2);
    ++b1;
    ++b2;
  }
  return result;
}


/**
  this procedure can be use to multiply by a scalar any element of vector-like objects
  as a result we may obtain object of different type,
    multiplication of column of matrix and scalar gives vector
  result = v * s
*/
template<typename ResultType, typename Object, typename FactorType>
void multiplyObjectScalar(const Object& v,const FactorType& s, ResultType& result)
{
  typename Object::const_iterator b=v.begin(), e=v.end();
  typename ResultType::iterator i=result.begin();
  while(b!=e)
  {
    *i = (*b) * s;
    ++i;
    ++b;
  }
}

/**
  this procedure can be use to multiply by a scalar any element of vector-like objects
  as a result we may obtain object of different type,
    multiplication of column of matrix and scalar gives vector
  result = v * s
*/
template<typename ResultType, typename Object, typename FactorType>
ResultType multiplyObjectScalar(const Object& v,const FactorType& s)
{
  ResultType result(v.dimension(),true);
  multiplyObjectScalar(v,s,result);
  return result;
}


/**
  this procedure can be use to divide by a scalar any element of vector-like objects
  as a result we may obtain object of different type,
    dividing of column of matrix by scalar gives vector
  result = v / s
*/
template<typename ResultType, typename Object, typename FactorType>
ResultType divideObjectScalar(const Object& v,const FactorType& s)
{
  ResultType result(v.dimension(),true);
  typename Object::const_iterator b=v.begin(), e=v.end();
  typename ResultType::iterator i=result.begin();
  while(b!=e)
  {
    *i = (*b) / s;
    ++i;
    ++b;
  }
  return result;
}


/**
  this procedure can be use to add a scalar to any element of vector-like objects
  result[i] = v[i] + s
*/
template<typename ResultType, typename Object, typename FactorType>
ResultType addObjectScalar(const Object& v,const FactorType& s)
{
  ResultType result(v.dimension(),true);
  typename Object::const_iterator b=v.begin(), e=v.end();
  typename ResultType::iterator i=result.begin();
  while(b!=e)
  {
    *i = (*b) + s;
    ++i;
    ++b;
  }
  return result;
}

/**
  this procedure can be use to subtract a scalar from any element of vector-like objects
  result[i] = v[i] - s
*/
template<typename ResultType, typename Object, typename FactorType>
ResultType subtractObjectScalar(const Object& v,const FactorType& s)
{
  ResultType result(v.dimension(),true);
  typename Object::const_iterator b=v.begin(), e=v.end();
  typename ResultType::iterator i=result.begin();
  while(b!=e)
  {
    *i = (*b) - s;
    ++i;
    ++b;
  }
  return result;
}

/**
  this procedure realizes multiplication of matrix-like object by vector-like object
  result = m*v
*/
template<typename ResultType, typename MatrixType, typename VectorType>
void matrixByVector(const MatrixType& m,const VectorType& u, ResultType& result){
  typename ResultType::iterator b=result.begin(), e=result.end();
  typename MatrixType::const_iterator i = m.begin();
  while(b!=e)
  {
    typename ResultType::ScalarType x = capd::TypeTraits<typename ResultType::ScalarType>::zero();
    typename VectorType::const_iterator bv=u.begin(), be=u.end();
    while(bv!=be)
    {
      x += (*bv) * (*i);
      ++bv;
      ++i;
    }
    *b=x;
    ++b;
  }
}


/**
  This procedure computes
  u += A*v
  where u,v are vectors and A is a matrix
*/
template<typename V, typename M, typename V2>
V& addAssignMatrixByVector(V& u, const M& A, const V2& v)
{
  typename V::iterator b=u.begin(), e=u.end();
  typename M::const_iterator i = A.begin();
  while(b!=e)
  {
    typename V2::const_iterator bv=v.begin(), be=v.end();
    while(bv!=be)
    {
      (*b) += (*bv) * (*i);
      ++bv;
      ++i;
    }
    ++b;
  }
  return u;
}

/**
  This procedure computes
  u -= A*v
  where u,v are vectors and A is a matrix
*/
template<typename V, typename M, typename V2>
V& subtractAssignMatrixByVector(V& u, const M& A, const V2& v)
{
  typename V::iterator b=u.begin(), e=u.end();
  typename M::const_iterator i = A.begin();
  while(b!=e)
  {
    typename V2::const_iterator bv=v.begin(), be=v.end();
    while(bv!=be)
    {
      (*b) -= (*bv) * (*i);
      ++bv;
      ++i;
    }
    ++b;
  }
  return u;
}

/**
  this procedure realizes multiplication of two matrix-like objects
  result = m1*m2
*/
template<typename ResultType, typename Matrix1, typename Matrix2>
void matrixByMatrix(const Matrix1& a1, const Matrix2& a2, ResultType& result)
{
  typedef typename ResultType::size_type size_type;
  for(size_type i=0;i<result.numberOfColumns();++i)
  {
    typename ResultType::iterator b=result.begin()+i, e=result.end()+i;
    typename Matrix1::const_iterator j=a1.begin();
    while(b!=e)
    {
      typename Matrix2::const_iterator b1=a2.begin()+i, e1=a2.end()+i;
      typename ResultType::ScalarType x = TypeTraits<typename ResultType::ScalarType>::zero();
      while(b1!=e1)
      {
        x += (*j) * (*b1);
        ++j;
        b1+=a2.rowStride();
      }
      *b=x;
      b+=result.rowStride();
    }
  }
}

//----------------------assignments - objects---------------------------//

/**
  this procedure can be use to assign one vector-like objects
  from the other.
  u = v
*/
template<typename T1, typename T2>
T1& assignObjectObject(T1& u, const T2& v){
  typename T1::iterator b=u.begin(), e=u.end();
  typename T2::const_iterator i = v.begin();
  while(b!=e)
  {
    *b = static_cast<typename T1::ScalarType>(*i);
    ++b;
    ++i;
  }
  return u;
}

/**
  this procedure can be use to add of two vector-like objects
  result is stored in the first argument
  u += v
*/
template<typename T1, typename T2>
T1& addAssignObjectObject(T1& u, const T2& v)
{
  if(u.dimension()!=v.dimension())
    throw std::range_error("capd::vectalg::addAssignObjectObject: Incompatible dimensions");
  typename T1::iterator i=u.begin();
  typename T2::const_iterator b=v.begin(), e=v.end();
  while(b!=e)
  {
    *i += *b;
    ++i;
    ++b;
  }
  return u;
}


/**
  this procedure can be use to subtract of two vector-like objects
  result is stored in the first argument
  u += v
*/
template<typename T1, typename T2>
T1& subtractAssignObjectObject(T1& u, const T2& v)
{
  if(u.dimension()!=v.dimension())
    throw std::range_error("capd::vectalg::subtractAssignObjectObject: Incompatible dimensions");
  typename T1::iterator i=u.begin();
  typename T2::const_iterator b=v.begin(), e=v.end();
  while(b!=e)
  {
    *i -= *b;
    ++i;
    ++b;
  }
  return u;
}

//----------------------assignments - Scalars---------------------------//

/**
  this procedure can be use to assign any element of two vector-like objects
  to be equal to a given scalar
  u[i] = s
*/
template<typename Object,typename Scalar>
Object& assignFromScalar(Object& u, const Scalar& s)
{
  typename Object::iterator b=u.begin(), e=u.end();
  while(b!=e)
  {
    *b = s;
    ++b;
  }
  return u;
}


/**
  this procedure can be use to add a scalar to any element of two vector-like objects
  u[i] += s
*/
template<typename Object,typename Scalar>
Object& addAssignObjectScalar(Object& u, const Scalar& s)
{
  typename Object::iterator b=u.begin(), e=u.end();
  while(b!=e)
  {
    *b += s;
    ++b;
  }
  return u;
}


/**
  this procedure can be use to subtract a scalar from any element of two vector-like objects
  u[i] -= s
*/
template<typename Object,typename Scalar>
Object& subtractAssignObjectScalar(Object& u, const Scalar& s)
{
  typename Object::iterator b=u.begin(), e=u.end();
  while(b!=e)
  {
    *b -= s;
    ++b;
  }
  return u;
}


/**
  this procedure can be use to multiply by a scalar any element of two vector-like objects
  u[i] *= s
*/
template<typename Object,typename Scalar>
Object& multiplyAssignObjectScalar(Object& u, const Scalar& s)
{
  typename Object::iterator b=u.begin(), e=u.end();
  while(b!=e)
  {
    *b *= s;
    ++b;
  }
  return u;
}

/**
  this procedure can be use to multiply by a scalar each element of a vector-like object
  and then add compoent-wise elements of second vector-like object
  u[i] = u[i]*s+v[i]
*/
template<typename Object, typename Object2, typename Scalar>
Object& multiplyAssignObjectScalarAddObject(Object& u, const Scalar& s, const Object2& v)
{
  typename Object::iterator b=u.begin(), e=u.end();
  typename Object2::const_iterator i = v.begin();
  while(b!=e)
  {
    *b = (*b)*s + (*i);
    ++b;
    ++i;
  }
  return u;
}


/**
  this procedure can be use to divide by a scalar any element of two vector-like objects
  u[i] /= s
*/
template<typename Object,typename Scalar>
Object& divideAssignObjectScalar(Object& u, const Scalar& s)
{
  typename Object::iterator b=u.begin(), e=u.end();
  while(b!=e)
  {
    *b /= s;
    ++b;
  }
  return u;
}
/**
 * Evaluates polynomial p:R->R^n at some argument.
 * Coefficients of the polynomial are assumed to be stored as an array c[r],
 * where c\in R^n. The result is stored in the third argument.
 *
 * @param c - coefficients of the polynomial
 * @param t - argument at which polynomial is evaluated
 * @param n - deree of the polynomial
 */
template<class Object, class Scalar>
void evalPolynomial(Object* c, Scalar t, Object& result, int n){
  result = c[n];
  for(int r = n - 1; r >= 0; --r)
    capd::vectalg::multiplyAssignObjectScalarAddObject(result,t,c[r]);
}


//-------coord-wise inequalities - true if true for each coord---------//

template<typename T1,typename T2>
bool lessThan (const T1& v1, const T2& v2)
{
  if(v1.dimension()!=v2.dimension())
    throw std::range_error("capd::vectalg::lessThan: Incompatible dimensions");
  typename T1::const_iterator b1=v1.begin(), e1=v1.end();
  typename T2::const_iterator b2=v2.begin();

  while(b1!=e1)
  {
    if(!(*b1 < *b2))
       return false;
    ++b1;
    ++b2;
  }
  return true;
}


template<typename T1,typename T2>
bool greaterThan (const T1& v1, const T2& v2)
{
  if(v1.dimension()!=v2.dimension())
    throw std::range_error("capd::vectalg::greaterThan: Incompatible dimensions");
  typename T1::const_iterator b1=v1.begin(), e1=v1.end();
  typename T2::const_iterator b2=v2.begin();

  while(b1!=e1)
  {
    if(!(*b1 > *b2))
       return false;
    ++b1;
    ++b2;
  }
  return true;
}


template<typename T1,typename T2>
bool lessEqual (const T1& v1, const T2& v2)
{
  if(v1.dimension()!=v2.dimension())
    throw std::range_error("capd::vectalg::lessEqual: Incompatible dimensions");
  typename T1::const_iterator b1=v1.begin(), e1=v1.end();
  typename T2::const_iterator b2=v2.begin();

  while(b1!=e1)
  {
    if(!(*b1 <= *b2))
       return false;
    ++b1;
    ++b2;
  }
  return true;
}


template<typename T1,typename T2>
bool greaterEqual (const T1& v1, const T2& v2)
{
  if(v1.dimension()!=v2.dimension())
    throw std::range_error("capd::vectalg::greaterEqual: Incompatible dimensions");
  typename T1::const_iterator b1=v1.begin(), e1=v1.end();
  typename T2::const_iterator b2=v2.begin();

  while(b1!=e1)
  {
    if(!(*b1 >= *b2))
       return false;
    ++b1;
    ++b2;
  }
  return true;
}

// ---------------------------------------- equality and not equality --------------------

template<typename T1,typename T2>
bool equal (const T1& v1, const T2& v2)
{
  if(v1.dimension()!=v2.dimension())
    throw std::range_error("capd::vectalg::equal: Incompatible dimensions");
  typename T1::const_iterator b1=v1.begin(), e1=v1.end();
  typename T2::const_iterator b2=v2.begin();

  while(b1!=e1)
  {
    if(!(*b1 == *b2))
       return false;
    ++b1;
    ++b2;
  }
  return true;
}


}} // end of namespace capd::vectalg

#endif // _CAPD_VECTALG_ALGEBRAICOPERATIONS_HPP_

/// @}

// --- End include/capd/vectalg/algebraicOperations.hpp ---

// --- Begin include/capd/vectalg/Vector.h ---
/////////////////////////////////////////////////////////////////////////////
/// @file Vector.h
///
/// This file provides a template class Vector together with typical
/// algebraic operations. Most of them are defined as generic algorithms
/// in files 'commonOperations.h' and 'commonOperations.hpp'
/// For inline definitions of operators see to file
/// Vector_inline.h included at the end of this file.
///
/// The class uses class 'Container' as a container for coefficients
///
/// @author The CAPD Group
/////////////////////////////////////////////////////////////////////////////

// Copyright (C) 2000-2013 by the CAPD Group.
//
// This file constitutes a part of the CAPD library,
// distributed under the terms of the GNU General Public License.
// Consult  http://capd.ii.uj.edu.pl/ for details.


#ifndef _CAPD_VECTALG_VECTOR_H_
#define _CAPD_VECTALG_VECTOR_H_

#include <iostream>
#include <cstdlib>
#include <vector>
// #include "capd/basicalg/minmax.h"
// #include "capd/basicalg/power.h"
// #include "capd/vectalg/Container.h"
//#include "capd/settings/compilerSetting.h"

namespace capd{
namespace vectalg{

/// @addtogroup vectalg
/// @{


template<typename Scalar, __size_type dim>
class Vector;

template<typename Scalar, __size_type dim>
std::ostream& operator<<(std::ostream& out, const Vector<Scalar,dim>& v);

template<typename Scalar, __size_type dim>
std::istream& operator>>(std::istream& inp, Vector<Scalar,dim>& v);


//########################### Vector template ################################//

template<typename Scalar, __size_type dim>
class Vector : public Container<Scalar,dim>
{
public:
  typedef Scalar ScalarType;
  typedef Container<Scalar,dim> ContainerType;
  typedef typename ContainerType::iterator iterator;
  typedef typename ContainerType::const_iterator const_iterator;
  typedef Vector<Scalar,dim> VectorType;
  typedef __size_type size_type;
  typedef __difference_type difference_type;

  Vector(void){}
  explicit Vector(size_type a_dim); // for compatibility with heap-allocated specialization
  Vector(const Scalar& x, const Scalar& y,const Scalar& z);  //obsolete, requires dimension=3
  Vector(size_type,const ScalarType[]);
  explicit Vector(const char data[]);
  explicit Vector(const std::string & data);
  Vector(const Vector&);
  template<typename S,
          typename std::enable_if<std::is_convertible<S, ScalarType>::value && !std::is_same<S, ScalarType>::value, int>::type = 0 >
  explicit Vector(const Vector<S,dim>&);

  Vector(size_type,bool); // it does not insert zeros

  template<__size_type dataDim>
  Vector(const Scalar (&data)[dataDim]);

  template<typename Iterator>
  Vector(Iterator begin, Iterator end);

  Vector(Vector&& v) = default;
  Vector & operator=(Vector && v)= default;
  Vector(std::initializer_list<ScalarType> l);

  // assignments - vectors
  Vector& operator=  (const Vector& v);      //<assign a vector
  Vector& operator+= (const Vector& v);      //<add and assign a vector
  Vector& operator-= (const Vector& v);      //<subtract and assign a vector

  // assignments - Scalars
  Vector& operator=  (const Scalar& s);       //<assign a Scalar to each coordinate
  Vector& operator+= (const Scalar& s);       //<component-wise increase by a Scalar
  Vector& operator-= (const Scalar& s);       //<component-wise decrease by a Scalar
  Vector& operator*= (const Scalar& s);       //<scale by multiplying by Scalar
  Vector& operator/= (const Scalar& s);       //<scale by dividing by Scalar

  template<typename U>
  struct rebind {
      typedef Vector<U,dim> other;
  };

  size_type dimension() const {return ContainerType::size();}
  using ContainerType::begin;
  using ContainerType::end;
  using ContainerType::rbegin;
  using ContainerType::rend;
  using ContainerType::resize;
  using ContainerType::clear;

  // Euclidean norm
  ScalarType euclNorm(void) const;
  //if possible vector is normalized and true is returned. Otherwise false is returned.
  bool normalize(void);
  void sorting_permutation(typename rebind<int>::other& perm);

  const static size_type csDim = dim;
  static size_type degree() {return 0;} // required interface for DynSys
  static Vector* makeArray(size_type N, size_type _dim);
protected:
  using ContainerType::size;

}; // end of class Vector template

template<typename Vector>
std::string vectorToString( const Vector & v, int firstIndex = 0, int lastIndex = -1, int precision = -1);

template<typename Vector>
std::ostream & printVector(std::ostream & str, const Vector & v, int firstIndex = 0, int lastIndex = -1);

template<typename Scalar, __size_type dim>
inline std::ostream & print(std::ostream & str, const Vector<Scalar, dim> & v, int firstIndex = 0, int lastIndex = -1){
  return printVector(str, v, firstIndex, lastIndex);
}

/// It serializes a matrix - gives text reprezentation which can be compiled
template<typename Scalar, __size_type dim>
std::string cppReprezentation(const Vector<Scalar,dim> & A, const std::string& varName,
			      const std::string& typeName);

/// @}


}} // namespace capd::vectalg

// #include "capd/vectalg/Vector_inline.h"

#endif // _CAPD_VECTALG_VECTOR_H_

// --- End include/capd/vectalg/Vector.h ---

// --- Begin include/capd/vectalg/Vector.hpp ---
/////////////////////////////////////////////////////////////////////////////
/// @file Vector.hpp
///
/// @author Marian Mrozek, Daniel Wilczak
/////////////////////////////////////////////////////////////////////////////

// Copyright (C) 2000-2013 by the CAPD Group.
//
// This file constitutes a part of the CAPD library,
// distributed under the terms of the GNU General Public License.
// Consult  http://capd.ii.uj.edu.pl/ for details.

#ifndef _CAPD_VECTALG_VECTOR_HPP_
#define _CAPD_VECTALG_VECTOR_HPP_

#include <cmath>
#include <stack>
#include <stdexcept>
#include <cstdio>
#include <sstream>
#include <algorithm>
// #include "capd/basicalg/minmax.h"
// #include "capd/basicalg/power.h"
// #include "capd/vectalg/Container.hpp"
// #include "capd/vectalg/Vector.h"
// #include "capd/vectalg/algebraicOperations.hpp"

namespace capd{
namespace vectalg{

//---------------------------constructors---------------------------//

template<typename Scalar,__size_type dim>
Vector<Scalar,dim>::Vector(size_type A_dimension, const ScalarType data[]) : ContainerType(A_dimension,true)
{
  std::copy(data,data+A_dimension,begin());
}

template<typename Scalar,__size_type dim>
Vector<Scalar,dim>::Vector(const char data[]) : ContainerType(dim,true)
{
   std::istringstream str(data);
   str >> *this;
}

template<typename Scalar,__size_type dim>
Vector<Scalar,dim>::Vector(const std::string & data) : ContainerType(dim,true)
{
   std::istringstream str(data);
   str >> *this;
}

template<typename Scalar,__size_type dim>
inline void setDimensionInArray(Vector<Scalar,dim>* , __size_type, __size_type){
}

template<typename Scalar>
inline void setDimensionInArray(Vector<Scalar,0>* t, __size_type N, __size_type _dim){
  for(__size_type i=0;i<N;++i) t[i].resize(_dim);
}

template<typename Scalar,__size_type dim>
Vector<Scalar,dim>* Vector<Scalar,dim>::makeArray(size_type N, size_type _dim)
{
  Vector* result = new Vector[N];
  setDimensionInArray(result,N,_dim);
  return result;
}

template<typename Scalar,__size_type dim>
void Vector<Scalar,dim>::sorting_permutation(typename rebind<int>::other& perm)
{
  typedef typename rebind<int>::other IntVectorType;
  difference_type i=0,j,k;

  if(dimension()!= perm.dimension())
     throw std::range_error("sorting_permutation: Incompatible vector dimensions");
  typename IntVectorType::iterator b=perm.begin(), e=perm.end();
  while(b!=e)
  {
    *b=i;
    ++i;
    ++b;
  }

  difference_type d = dimension();

  for(i=0;i<d;i++)
    for(j=d-1;j>i;j--)
    {
      if((*this)[perm[j]] > (*this)[perm[j-1]])
      {
        k=perm[j-1];
        perm[j-1]=perm[j];
        perm[j]=k;
      }
    }
}

template<typename Scalar,__size_type dim>
template<__size_type dataDim>
Vector<Scalar,dim>::Vector(const Scalar (&data)[dataDim]): ContainerType(dataDim,true)
{
  std::copy(data, data + dataDim, begin());
}

template<typename Scalar,__size_type dim>
template<typename Iterator>
Vector<Scalar,dim>::Vector(Iterator begin, Iterator end): ContainerType((end - begin), true)
{
  std::copy(begin, end, this->begin());
}

//----------------- input-output ----------------------------------//

template<typename Scalar,__size_type dim>
std::ostream& operator<<(std::ostream& out, const Vector<Scalar,dim>& v)
{
  typedef typename Vector<Scalar,dim>::size_type size_type;
  const size_type d = v.dimension();
  out << "{";
  if(d>0){
     //if(v[0]>=Scalar(0)) out << " "; /***** DW it does not work for complex vectors ***/
     out << v[0];
   }
   for(size_type i=1;i<d;i++)
   {
      out << ",";
      // if(v[i]>=Scalar(0)) out << " "; /***** DW it does not work for complex vectors ***/
      out << v[i];
   }
   out << "}";
   return out;
}

template<typename Vector>
std::string vectorToString( const Vector & v, int firstIndex /*= 0*/, int lastIndex /*= -1*/,  int precision /* = -1*/){
  std::ostringstream out;
  if(precision>0)
       out.precision(precision);
  print(out, v, firstIndex, lastIndex);
  return out.str();
}

template<typename Vector>
std::ostream & printVector(std::ostream & out, const Vector & v, int firstIndex /*= 0*/, int lastIndex /*= -1*/){

  const int d = v.dimension();

  if((lastIndex < 0) || (lastIndex >= d))
    lastIndex = d-1;

  if(firstIndex < d) {
    if(firstIndex < 0)
      firstIndex = 0;
    out << "{" << v[firstIndex];
    for(int i=firstIndex+1;i<=lastIndex;i++) {
      out << "," << v[i];
    }
    out << "}";
  } else {
    out << "{}";
  }
  return out;
}

template<typename Scalar,__size_type dim>
std::istream& operator>> (std::istream& inp, Vector<Scalar,dim>& v)
{
   std::deque<Scalar> st;
   Scalar s;
   int ch;

   while('{'!=(ch=inp.get()) && ch!=EOF)
     ;
   if(ch!= EOF)
   {
/*
      // -- begin of added lines for empty vectors
      while(' '==(ch=inp.peek())) ch=inp.get();
      if('}'==(ch=inp.peek())){
        ch=inp.get();
        return inp;
      }
      // -- end of added lines for empty vectors
*/
      inp >> s;
      st.push_back(s);
      do{
         do{
            ch=inp.get();
         }while(isspace(ch));
         if(ch==','){
            inp >> s;
            st.push_back(s);
         }
      }while(ch!='}' && ch!=EOF);
   }
   if(inp.eof())
       throw std::ios_base::failure("EOF encountered when reading a vector");
   v.resize(st.size());
   std::copy(st.begin(), st.end(), v.begin());
   return inp;
}

template<typename Scalar, __size_type dim>
std::string cppReprezentation(const Vector<Scalar,dim> & A, const std::string& varName,
			      const std::string& typeName)
{
  std::stringstream out;
  out << "capd::vectalg::Vector<" << typeName << ", " << dim << "> " << varName << "(";

  if (A.dimension() > 0) {
    out << "(" << typeName << "[" << A.dimension() << "])" << A;
  } else {
    out << "(capd::vectalg::__size_type)" << A.dimension();
  }

  out << ");";

  std::string str = out.str();
  std::replace(str.begin(), str.end(), '\n', ' ');

  return str;
}

}} // namespace capd::vectalg

#endif // _CAPD_VECTALG_VECTOR_HPP_

// --- End include/capd/vectalg/Vector.hpp ---

// --- Begin include/capd/vectalg/Vector_inline.h ---
/////////////////////////////////////////////////////////////////////////////
/// @file Vector_inline.h
///
/// This file contains inline definitions for class Vector
///
/// @author The CAPD Group
/////////////////////////////////////////////////////////////////////////////

// Copyright (C) 2000-2013 by the CAPD Group.
//
// This file constitutes a part of the CAPD library,
// distributed under the terms of the GNU General Public License.
// Consult  http://capd.ii.uj.edu.pl/ for details.


#ifndef _CAPD_VECTALG_VECTOR_INLINE_H_
#define _CAPD_VECTALG_VECTOR_INLINE_H_

// #include "capd/vectalg/Vector.h"
// #include "capd/vectalg/algebraicOperations.h"

namespace capd{
namespace vectalg{

template<typename Scalar, __size_type dim>
class Vector;

template<typename Scalar, __size_type dim>
inline Vector<Scalar,dim> abs(const Vector<Scalar,dim>& v){
  return absoluteValue< Vector<Scalar,dim>, Vector<Scalar,dim> > (v);
}

template<typename Scalar, __size_type dim>
inline Vector<Scalar,dim> operator-(const Vector<Scalar,dim>& v){
  return unaryMinus< Vector<Scalar,dim>, Vector<Scalar,dim> >(v);
}

template<typename Scalar, __size_type dim>
inline Vector<Scalar,dim> operator+(const Vector<Scalar,dim>& v1,const Vector<Scalar,dim>& v2){
  return addObjects< Vector<Scalar,dim>, Vector<Scalar,dim>, Vector<Scalar,dim> >(v1,v2);
}

template<typename Scalar, __size_type dim>
inline Vector<Scalar,dim> operator-(const Vector<Scalar,dim>& v1,const Vector<Scalar,dim>& v2){
  return subtractObjects< Vector<Scalar,dim>, Vector<Scalar,dim>, Vector<Scalar,dim> >(v1,v2);
}

template<typename Scalar, __size_type dim>
inline Scalar operator*(const Vector<Scalar,dim>& v1,const Vector<Scalar,dim>& v2){
  return scalarProduct< Vector<Scalar,dim>, Vector<Scalar,dim> >(v1,v2);
}

template<typename Scalar, __size_type dim>
inline Vector<Scalar,dim> operator+(const Vector<Scalar,dim>& v, const Scalar& s){
  return addObjectScalar< Vector<Scalar,dim>, Vector<Scalar,dim>, Scalar >(v,s);
}

template<typename Scalar, __size_type dim>
inline Vector<Scalar,dim> operator-(const Vector<Scalar,dim>& v,const Scalar& s){
  return subtractObjectScalar< Vector<Scalar,dim>, Vector<Scalar,dim>, Scalar >(v,s);
}


template<typename Scalar, typename FactorType, __size_type dim>
inline Vector<Scalar,dim> operator*(const Vector<Scalar,dim>& v, const FactorType& s){
  return multiplyObjectScalar< Vector<Scalar,dim>, Vector<Scalar,dim>, FactorType >(v,s);
}

template<typename Scalar, typename FactorType, __size_type dim>
inline Vector<Scalar,dim> operator*(const FactorType& s,const Vector<Scalar,dim>& v){
  return multiplyObjectScalar< Vector<Scalar,dim>, Vector<Scalar,dim>, FactorType >(v,s);
}

template<typename Scalar, typename FactorType, __size_type dim>
Vector<Scalar,dim> operator/(const Vector<Scalar,dim>& v, const FactorType& s){
  return divideObjectScalar< Vector<Scalar,dim>, Vector<Scalar,dim>, FactorType >(v,s);
}


template<typename Scalar,__size_type dim>
inline Vector<Scalar,dim>& Vector<Scalar,dim>::operator=(const Vector<Scalar,dim>& v){
  ContainerType::operator=(v);
  return *this;
}

template<typename Scalar,__size_type dim>
inline Vector<Scalar,dim>::Vector(size_type A_dimension) : ContainerType(A_dimension)
{}

template<typename Scalar,__size_type dim>
inline Vector<Scalar,dim>::Vector(const Vector& A_vect) : ContainerType(A_vect)
{}

template<typename Scalar,__size_type dim>
inline Vector<Scalar,dim>::Vector(const Scalar& x,const Scalar& y,const Scalar& z) : ContainerType(3,true){
  (*this)[0]=x; (*this)[1]=y; (*this)[2]=z;
}

template<typename Scalar,__size_type dim>
inline Vector<Scalar,dim>::Vector(size_type a_Dimension,bool) : ContainerType(a_Dimension,true)
{}

//template<typename Scalar,__size_type dim>
//inline Vector<Scalar,dim>::Vector(Vector&& v) : ContainerType(std::move(v)) {
//}
//template<typename Scalar,__size_type dim>
//inline Vector<Scalar,dim> & Vector<Scalar,dim>::operator=(Vector && v) {
//  ContainerType::operator=(std::move(v));
//  return *this;
//}

template<typename Scalar,__size_type dim>
inline Vector<Scalar,dim>::Vector(std::initializer_list<ScalarType> l)
  : ContainerType(l.size(), false) {
    if(l.size() == this->size())
		std::copy(l.begin(), l.end(), this->begin());
    else
      throw std::range_error("Constructor of Vector with static size got initializer list with wrong size.");
}

template<typename Scalar,__size_type dim>
inline Vector<Scalar,dim>& Vector<Scalar,dim>::operator+=(const Vector<Scalar,dim>& v){
   return addAssignObjectObject < Vector<Scalar,dim>, Vector<Scalar,dim> > (*this,v);
}


template<typename Scalar,__size_type dim>
inline Vector<Scalar,dim>& Vector<Scalar,dim>::operator-=(const Vector& v){
   return subtractAssignObjectObject < Vector<Scalar,dim>, Vector<Scalar,dim> > (*this,v);
}

template<typename Scalar,__size_type dim>
inline Vector<Scalar,dim>& Vector<Scalar,dim>::operator=(const Scalar& s){
   return assignFromScalar< Vector<Scalar,dim>, Scalar > (*this,s);
}


template<typename Scalar,__size_type dim>
inline Vector<Scalar,dim>& Vector<Scalar,dim>::operator+=(const Scalar& s){
  return addAssignObjectScalar< Vector<Scalar,dim>, Scalar > (*this,s);
}

template<typename Scalar,__size_type dim>
inline Vector<Scalar,dim>& Vector<Scalar,dim>::operator-=(const Scalar& s){
  return subtractAssignObjectScalar< Vector<Scalar,dim>, Scalar > (*this,s);
}

template<typename Scalar,__size_type dim>
inline Vector<Scalar,dim>& Vector<Scalar,dim>::operator*=(const Scalar& s){
  return multiplyAssignObjectScalar< Vector<Scalar,dim>, Scalar > (*this,s);
}

template<typename Scalar,__size_type dim>
inline Vector<Scalar,dim>& Vector<Scalar,dim>::operator/=(const Scalar& s){
  return divideAssignObjectScalar< Vector<Scalar,dim>, Scalar > (*this,s);
}


template<typename Scalar,__size_type dim>
inline bool operator< (const Vector<Scalar,dim>& v1, const Vector<Scalar,dim>& v2){
  return lessThan(v1,v2);
}


template<typename Scalar,__size_type dim>
inline bool operator> (const Vector<Scalar,dim>& v1, const Vector<Scalar,dim>& v2){
  return greaterThan(v1,v2);
}


template<typename Scalar,__size_type dim>
inline bool operator<= (const Vector<Scalar,dim>& v1, const Vector<Scalar,dim>& v2){
  return lessEqual(v1,v2);
}


template<typename Scalar,__size_type dim>
inline bool operator>= (const Vector<Scalar,dim>& v1, const Vector<Scalar,dim>& v2){
  return greaterEqual(v1,v2);
}

// ----------------------------------- equality --------------------------------------

template<typename Scalar,__size_type dim>
inline bool operator== (const Vector<Scalar,dim>& v1, const Vector<Scalar,dim>& v2){
  return equal(v1,v2);
}


template<typename Scalar,__size_type dim>
inline bool operator!= (const Vector<Scalar,dim>& v1, const Vector<Scalar,dim>& v2){
  return notEqual(v1,v2);
}

template<typename Scalar,__size_type dim>
inline typename Vector<Scalar,dim>::ScalarType Vector<Scalar,dim>::euclNorm(void) const{
  return capd::vectalg::euclNorm(*this);
}

template<typename Scalar,__size_type dim>
inline bool Vector<Scalar,dim>::normalize(){
  return capd::vectalg::normalize(*this);
}

template<typename Scalar,__size_type dim>
template<typename S,
        typename std::enable_if<std::is_convertible<S, Scalar>::value && !std::is_same<S, Scalar>::value , int>::type>
inline Vector<Scalar,dim>::Vector(const Vector<S,dim>& v) : ContainerType(v.dimension(),true){
  assignObjectObject(*this,v);
}

}} // namespace capd::vectalg

#endif // _CAPD_VECTALG_VECTOR_INLINE_H_


// --- End include/capd/vectalg/Vector_inline.h ---

// --- Begin include/capd/vectalg/ColumnVector.h ---
/// @addtogroup vectalg
/// @{

/////////////////////////////////////////////////////////////////////////////
/// @file ColumnVector.h
///
/// This file provides a template class ColumnVector
/// This class realizes a vector without own container, which is a reference
/// to a subset of other object with his own container.
/// A typical situation is a column of matrix which can be consider as a vector
///
/// The file 'RowVector.h' defines similar class, but in that case it is assumed
/// that data fill continuous part of a memory
///
/// @author Daniel Wilczak
/////////////////////////////////////////////////////////////////////////////

// Copyright (C) 2000-2005 by the CAPD Group.
//
// This file constitutes a part of the CAPD library,
// distributed under the terms of the GNU General Public License.
// Consult  http://capd.ii.uj.edu.pl/ for details.

#ifndef _CAPD_VECTALG_COLUMNVECTOR_H_
#define _CAPD_VECTALG_COLUMNVECTOR_H_

#include <ostream>
// #include "capd/vectalg/Vector.h"

namespace capd{
namespace vectalg{

template<typename Scalar>
class ColumnIterator{
public:
  typedef Scalar ScalarType;
  typedef __difference_type difference_type;

  inline ColumnIterator(ScalarType *p, difference_type pointerStride)
  : m_Pointer(p), m_pointerStride(pointerStride)
  {}

  inline ColumnIterator operator++(int)
  {
    ColumnIterator it(*this);
    m_Pointer += m_pointerStride;
    return it;
  }

  inline ColumnIterator operator--(int)
  {
    ColumnIterator it(*this);
    m_Pointer -= m_pointerStride;
    return it;
  }

  inline ColumnIterator& operator++()
  {
    m_Pointer += m_pointerStride;
    return *this;
  }

  inline ColumnIterator& operator--()
  {
    m_Pointer -= m_pointerStride;
    return *this;
  }

  inline ColumnIterator& operator+=(difference_type jump)
  {
    m_Pointer+= jump*m_pointerStride;
    return *this;
  }

  inline ColumnIterator& operator-=(difference_type jump)
  {
    m_Pointer-= jump*m_pointerStride;
    return *this;
  }

  inline bool operator!=(const ColumnIterator& second)
  {
    return m_Pointer!=second.m_Pointer;
  }

  inline ScalarType& operator*()
  {
    return *m_Pointer;
  }

  inline ScalarType* operator->()
  {
    return m_Pointer;
  }

private:
  ScalarType *m_Pointer;
  difference_type m_pointerStride;
  ColumnIterator(){} // we do not need a default constructor
};

// --------------------- const_iterator -------------------

template<typename Scalar>
class const_ColumnIterator{
public:
  typedef Scalar ScalarType;
  typedef __difference_type difference_type;

  inline const_ColumnIterator(const ScalarType* p, difference_type pointerStride)
    : m_Pointer(p), m_pointerStride(pointerStride)
  {}

  inline const_ColumnIterator operator++(int)
  {
    const_ColumnIterator it(*this);
    m_Pointer += m_pointerStride;
    return it;
  }

  inline const_ColumnIterator operator--(int)
  {
    const_ColumnIterator it(*this);
    m_Pointer -= m_pointerStride;
    return it;
  }

  inline const_ColumnIterator& operator++()
  {
    m_Pointer += m_pointerStride;
    return *this;
  }

  inline const_ColumnIterator& operator--()
  {
    m_Pointer -= m_pointerStride;
    return *this;
  }

  inline const_ColumnIterator& operator+=(difference_type jump)
  {
    m_Pointer += jump*m_pointerStride;
    return *this;
  }

  inline const_ColumnIterator& operator-=(difference_type jump)
  {
    m_Pointer -= jump*m_pointerStride;
    return *this;
  }

  inline bool operator!=(const const_ColumnIterator& second)
  {
    return m_Pointer!=second.m_Pointer;
  }

  inline const ScalarType& operator*()
  {
    return *m_Pointer;
  }

  inline const ScalarType* operator->()
  {
    return m_Pointer;
  }
private:
  const ScalarType*  m_Pointer;
  const_ColumnIterator(){} // we do not need a default constructor
  difference_type m_pointerStride;
};


template<typename Scalar, __size_type rows, __size_type cols>
class Matrix;

template<typename Scalar,__size_type dim>
class Vector;

/// This class realizes a vector without its own container, which is a reference
/// to a subset of other object with his own container.
/// A typical situation is a column of matrix which can be considered as a vector
///
template<typename Scalar, __size_type rows>
class ColumnVector
{
public:
  typedef Scalar ScalarType;
  typedef capd::vectalg::ColumnIterator<Scalar> iterator;
  typedef capd::vectalg::const_ColumnIterator<Scalar> const_iterator;
  typedef ColumnVector VectorType;
  typedef ColumnVector ContainerType;
  typedef __size_type size_type;
  typedef __difference_type difference_type;

  ColumnVector(const Scalar* pointer, difference_type stride, size_type dim);
  ColumnVector& operator=(const ColumnVector&);
  ColumnVector& operator=(const Vector<Scalar,rows>&);
  ColumnVector& operator+=(const ColumnVector&);
  ColumnVector& operator+=(const Vector<Scalar,rows>&);
  ColumnVector& operator-=(const ColumnVector&);
  ColumnVector& operator-=(const Vector<Scalar,rows>&);
  ColumnVector& operator*=(const Scalar&);
  ColumnVector& operator/=(const Scalar&);
  operator Vector<Scalar,rows>() const;

  Scalar& operator[](size_type row);
  const Scalar& operator[](size_type row) const;

  Scalar euclNorm() const;
  bool normalize();
  void clear();
  size_type dimension() const;
  void next();

  iterator begin();
  iterator end();
  const_iterator begin() const;
  const_iterator end() const;

  void assertEqualSize(const ColumnVector& c) const;
protected:
  Scalar* m_pointer;
  difference_type m_stride;
  size_type m_dim;
}; // end of class ColumnVector

// -------------------------- inline definitions ------------------------

template<typename Scalar, __size_type rows>
inline std::ostream& operator << (std::ostream& out, const ColumnVector<Scalar,rows>& s){
  return out << Vector<Scalar,rows>(s);
}

template<typename Scalar, __size_type rows>
inline Scalar ColumnVector<Scalar,rows>::euclNorm() const{
  return capd::vectalg::euclNorm(*this);
}

template<typename Scalar, __size_type rows>
inline bool ColumnVector<Scalar,rows>::normalize(){
  return capd::vectalg::normalize(*this);
}

template<typename Scalar, __size_type rows>
inline ColumnVector<Scalar,rows>& ColumnVector<Scalar,rows>::operator=(const ColumnVector& v){
  return assignObjectObject(*this,v);
}

template<typename Scalar, __size_type rows>
inline ColumnVector<Scalar,rows>& ColumnVector<Scalar,rows>::operator=(const Vector<Scalar,rows>& v){
  return assignObjectObject(*this,v);
}

// ----------------------- iterator selection ---------------------------

template<typename Scalar, __size_type rows>
inline typename ColumnVector<Scalar,rows>::size_type
ColumnVector<Scalar,rows>::dimension() const{
  return m_dim;
}

template<typename Scalar, __size_type rows>
inline typename ColumnVector<Scalar,rows>::iterator
ColumnVector<Scalar,rows>::begin(){
  return iterator(m_pointer,m_stride);
}

template<typename Scalar, __size_type rows>
inline typename ColumnVector<Scalar,rows>::iterator
ColumnVector<Scalar,rows>::end(){
  return iterator(m_pointer+m_dim*m_stride, m_stride);
}

template<typename Scalar, __size_type rows>
inline typename ColumnVector<Scalar,rows>::const_iterator
ColumnVector<Scalar,rows>::begin() const{
  return const_iterator(m_pointer, m_stride);
}

template<typename Scalar, __size_type rows>
inline typename ColumnVector<Scalar,rows>::const_iterator
ColumnVector<Scalar,rows>::end() const{
  return const_iterator(m_pointer+m_dim*m_stride, m_stride);
}

// ------------------------------ resize -----------------------------------

template<typename Scalar, __size_type rows>
inline void ColumnVector<Scalar,rows>::assertEqualSize(const ColumnVector& c) const{
  if(m_dim!=c.dimension())
    throw std::runtime_error("Unequal dimensions in ColumnVector::assertEqualSize");
}

// ------------------------------ constructor -----------------------------

template<typename Scalar, __size_type rows>
inline ColumnVector<Scalar,rows>::ColumnVector(const Scalar* pointer, difference_type stride, size_type dim)
    : m_pointer(const_cast<Scalar*>(pointer)),
      m_stride(stride), m_dim(dim)
{}

template<typename Scalar, __size_type rows>
inline Scalar& ColumnVector<Scalar,rows>::operator[](size_type row){
  return *(m_pointer + row*m_stride);
}

template<typename Scalar, __size_type rows>
inline const Scalar& ColumnVector<Scalar,rows>::operator[](size_type row) const{
  return *(m_pointer + row*m_stride);
}

template<typename Scalar, __size_type rows>
void ColumnVector<Scalar,rows>::next(){
  m_pointer++;
}

// -------------------- operator + ------------------------------------------

template<typename Scalar, __size_type rows>
inline Vector<Scalar,rows> operator+(const Vector<Scalar,rows>& u,
                                       const ColumnVector<Scalar,rows>& v
                                      )
{
  return addObjects< Vector<Scalar,rows>, Vector<Scalar,rows>, ColumnVector<Scalar,rows> > (u,v);
}

template<typename Scalar, __size_type rows>
inline Vector<Scalar,rows> operator+(const ColumnVector<Scalar,rows>&v,
                                       const Vector<Scalar,rows>&u
                                      )
{
  return addObjects< Vector<Scalar,rows>, Vector<Scalar,rows>, ColumnVector<Scalar,rows> > (u,v);
}

template<typename Scalar, __size_type rows>
inline Vector<Scalar,rows> operator+(const ColumnVector<Scalar,rows>&u,
                                       const ColumnVector<Scalar,rows>&v
                                      )
{
  return addObjects< Vector<Scalar,rows>, ColumnVector<Scalar,rows>, ColumnVector<Scalar,rows> > (u,v);
}

// -------------------- operator - ------------------------------------------

template<typename Scalar, __size_type rows>
inline Vector<Scalar,rows> operator-(const Vector<Scalar,rows>& u,
                                       const ColumnVector<Scalar,rows>& v
                                      )
{
  return subtractObjects< Vector<Scalar,rows>, Vector<Scalar,rows>, ColumnVector<Scalar,rows> > (u,v);
}

template<typename Scalar, __size_type rows>
inline Vector<Scalar,rows> operator-(const ColumnVector<Scalar,rows>&v,
                                       const Vector<Scalar,rows>&u
                                      )
{
  return subtractObjects< Vector<Scalar,rows>, Vector<Scalar,rows>, ColumnVector<Scalar,rows> > (u,v);
}

template<typename Scalar, __size_type rows>
inline Vector<Scalar,rows> operator-(const ColumnVector<Scalar,rows>&u,
                                       const ColumnVector<Scalar,rows>&v
                                      )
{
  return subtractObjects< Vector<Scalar,rows>, ColumnVector<Scalar,rows>, ColumnVector<Scalar,rows> > (u,v);
}

// ------------------------------- scalar product ----------------------------

template<typename Scalar, __size_type rows>
inline Scalar operator*(const Vector<Scalar,rows>& u, const ColumnVector<Scalar,rows>& v){
  return scalarProduct(u,v);
}

template<typename Scalar, __size_type rows>
inline Scalar operator*(const ColumnVector<Scalar,rows>&v, const Vector<Scalar,rows>&u){
  return scalarProduct(u,v);
}

template<typename Scalar, __size_type rows>
inline Scalar operator*(const ColumnVector<Scalar,rows>&v, const ColumnVector<Scalar,rows>&u){
  return scalarProduct(u,v);
}

// ------------------------- unary minus ----------------------------------------

template<typename Scalar, __size_type rows>
inline Vector<Scalar,rows> operator-(const ColumnVector<Scalar,rows>&u){
  return unaryMinus< Vector<Scalar,rows>, ColumnVector<Scalar,rows> >(u);
}

// -------------------------- multiplication and division by scalar -------------

template<typename Scalar, __size_type rows>
inline Vector<Scalar,rows> operator*(const Scalar&s, const ColumnVector<Scalar,rows>&u){
  return multiplyObjectScalar< Vector<Scalar,rows> > (u,s);
}

template<typename Scalar, __size_type rows>
inline Vector<Scalar,rows> operator*(const ColumnVector<Scalar,rows>&u, const Scalar&s){
  return multiplyObjectScalar< Vector<Scalar,rows> > (u,s);
}

template<typename Scalar, __size_type rows>
inline Vector<Scalar,rows> operator/(const ColumnVector<Scalar,rows>&u, const Scalar&s){
  return divideObjectScalar< Vector<Scalar,rows> > (u,s);
}

template<typename Scalar, __size_type rows>
inline ColumnVector<Scalar,rows>& ColumnVector<Scalar,rows>::operator*=(const Scalar& s){
  return multiplyAssignObjectScalar(*this,s);
}

template<typename Scalar, __size_type rows>
ColumnVector<Scalar,rows>& ColumnVector<Scalar,rows>::operator/=(const Scalar& s){
  return divideAssignObjectScalar(*this,s);
}

// -------------------------------------- assignments ---------------------------------------

template<typename Scalar, __size_type rows>
inline ColumnVector<Scalar,rows>& ColumnVector<Scalar,rows>::operator+=(const ColumnVector& v){
  return addAssignObjectObject(*this,v);
}

template<typename Scalar, __size_type rows>
inline ColumnVector<Scalar,rows>& ColumnVector<Scalar,rows>::operator+=(const Vector<Scalar,rows>& v){
  return addAssignObjectObject(*this,v);
}

template<typename Scalar, __size_type rows>
inline ColumnVector<Scalar,rows>& ColumnVector<Scalar,rows>::operator-=(const ColumnVector& v){
  return subtractAssignObjectObject(*this,v);
}

template<typename Scalar, __size_type rows>
inline ColumnVector<Scalar,rows>& ColumnVector<Scalar,rows>::operator-=(const Vector<Scalar,rows>& v){
  return subtractAssignObjectObject(*this,v);
}


template<typename Scalar, __size_type rows>
void ColumnVector<Scalar,rows>::clear(){
  capd::vectalg::clear(*this);
}



/// It serializes a matrix - gives text reprezentation which can be compiled
template<typename Scalar, __size_type rows>
std::string cppReprezentation(const ColumnVector<Scalar,rows> & A, const std::string& varName,
			      const std::string& typeName);

}} // namespace capd::vectalg

#endif // _CAPD_VECTALG_COLUMNVECTOR_H_

/// @}

// --- End include/capd/vectalg/ColumnVector.h ---

// --- Begin include/capd/vectalg/Multiindex.h ---
/// @addtogroup vectalg
/// @{

/////////////////////////////////////////////////////////////////////////////
/// @file Multiindex.h
///
/// @author Daniel Wilczak
/////////////////////////////////////////////////////////////////////////////

// Copyright (C) 2000-2005 by the CAPD Group.
//
// This file constitutes a part of the CAPD library,
// distributed under the terms of the GNU General Public License.
// Consult  http://capd.ii.uj.edu.pl/ for details.

#ifndef _CAPD_VECTALG_MULTIINDEX_H_
#define _CAPD_VECTALG_MULTIINDEX_H_

#include <stdexcept>
#include <vector>
// #include "capd/basicalg/factrial.h"
// #include "capd/basicalg/TypeTraits.h"
// #include "capd/vectalg/Vector.h"

namespace capd{
namespace vectalg{


class Multiindex;


/// Multipointer always contains nondecreasing list of indexes of variables.
///
/// For partial derivatives they denote variables with respect to which we differentiate.
/// For example, a Multipointer mp=(0,0,2,3) corresponds to partial derivative
/// \f$ \frac{\partial^4}{\partial x_0^2 \partial x_2 \partial x_3} \f$
///
/// For power series multipointer denotes variables that form monomial.
/// e.g.Multipointer mp=(0,0,2,3) correspond to monomial \f$ x_0^2x_2x_3\f$
///
class Multipointer : public Vector<int,0>
{
public:
  typedef Vector<int,0>::iterator iterator;
  typedef Vector<int,0>::const_iterator const_iterator;
  typedef Vector<int,0>::size_type size_type;

  Multipointer(void){}
  explicit Multipointer(size_type _dimension) : Vector<int,0>(_dimension){}
  explicit Multipointer(const Multiindex&);
  Multipointer(const Multipointer& p) : Vector<int,0>(p) {}
  Multipointer(size_type dim, int data[]) : Vector<int,0>(dim,data){}
  Multipointer(size_type dim,bool) : Vector<int,0>(dim,true){}
  Multipointer& operator=(const Multipointer& v) {
    Vector<int,0>::operator= ( static_cast< const Vector<int,0> &>(v));
    return *this;
  }

  Multipointer(Multipointer&& v) : Vector<int,0>(std::move(v)) {}
  Multipointer & operator=(Multipointer && v) {
    Vector<int,0>::operator= ( static_cast< Vector<int,0> &&>(v));
    return *this;
  }
  Multipointer(std::initializer_list<int> l) : Vector<int,0>(l.size(), false) {
    std::copy(l.begin(), l.end(), this->begin());
  }

  /// order of derivative
  inline size_type module() const{
    return dimension();
  }
  bool hasNext(size_type dim);
  long factorial() const;
  Multipointer subMultipointer(const Multipointer& mp) const;

  typedef std::vector<Multipointer> MultipointersVector;
  typedef std::vector<MultipointersVector> IndicesSet;
  static const IndicesSet& generateList(size_type p, size_type k);

  size_type index(size_type dimension, size_type maxDegree) const;
  size_type index(size_type dimension, size_type maxDegree, const Multipointer& sub) const;

private:
  static std::vector<IndicesSet> indices;
  static void computeNextLevel();
  static size_type maxKnownLevel;
  inline static IndicesSet& getList(size_type n, size_type k)
  {
    return indices[n*(n-1)/2+k-1];
  }
};

// -------------------------------------------------------------------------------

/// For a Multiindex mi, mi[p] is a number of differentiation with respect to i-th variable.
/// For example, a Multipointer mp=(0,0,2,3) in 5-dimensional space corresponds to
/// the Multiindex mi=(2,0,1,1,0).
/// Hence, Multiindex agrees with standard notation and it contains an additional information
/// about the dimension of the domain of the function.
///
/// For polynomial:  Multiindex stores exponents of a given monomial.
/// e.g. monomial \f$ x^2 z^3 \f$ of 4 variables (x,y,z,t) has multiindex (2,0,3,0)
///
class Multiindex : public Vector<int,0>
{
public:
  typedef Vector<int,0>::iterator iterator;
  typedef Vector<int,0>::const_iterator const_iterator;
  typedef std::vector<Multiindex> MultiindexVector;
  typedef std::vector<MultiindexVector> IndicesSet;
  typedef Vector<int,0>::size_type size_type;

  Multiindex(void){}
  explicit Multiindex(size_type _dimension) : Vector<int,0>(_dimension){}
  Multiindex(size_type _dimension, const Multipointer&);
  Multiindex(size_type dim, int data[]) : Vector<int,0>(dim,data){}
  Multiindex(size_type dim,bool) : Vector<int,0>(dim,true){}
  Multiindex(const Multiindex& v) : Vector<int,0>(v) {}
  Multiindex& operator=(const Multiindex & v) {
    Vector<int,0>::operator= ( static_cast< const Vector<int,0> &>(v));
    return *this;
  }


  Multiindex(Multiindex&& v) : Vector<int,0>(std::move(v)) {}
  Multiindex & operator=(Multiindex && v) {
    Vector<int,0>::operator= ( static_cast< Vector<int,0> &&>(v));
    return *this;
  }
  Multiindex(std::initializer_list<int> l) : Vector<int,0>(l.size(), false) {
    std::copy(l.begin(), l.end(), this->begin());
  }

  bool hasNext();
  bool hasNext(int* a, int* b) const;
  bool hasNext(int* a, int* b, size_type j) const;

  size_type module() const;          ///< returns sum of multiindex coordinates (i.e. degree of monomial)
  long factorial() const;      ///< for multiindex (a,b,..,n) returns a!b!...n!
  static void generateList(size_type n, size_type k, IndicesSet& result);
  size_type index(size_type maxDegree) const; ///< computes index in the array that corresponds to this multiindex
};

// -------------------------------------------------------------------------------

Multipointer sumMultipointers(const Multipointer&, const Multipointer&);

/// returns new multipointer which is multiindex mp with index added in correct place
Multipointer addIndex(const Multipointer & mp, int index);

/// appends index to the end of multipointer mp
Multipointer push_back(const Multipointer & mp, int index);

/**
  checks if muiltipointer contains index
  @param[in] - mp multipointer that is checked
  @param[in] - index index to be found
  @return true if mp contains index
*/
bool hasIndex(const  Multipointer & mp, int index);

/**
  returns the number of occurences of index in the multipointer
  @param[in] mp - multipointer that is inspected
  @param[in] index - the counted index
  @return number of occureneces o of index in mp
*/
int indexCount(const Multipointer &mp, int index);

/**
  returns a multipointer with removed index
  @param[in] mp - multipointer from which an index is removed
  @param[in] index -index to be removed
  @returns a multipointer with removed index
 */
Multipointer removeIndex(const Multipointer & mp, int index);

}} // namespace capd::vectalg

// -------------------------------------------------------------------------------

///
/// It computes v^m where v is a vector and m is a multiindex
///

template<typename VectorType>
typename VectorType::ScalarType power(const VectorType& v, const capd::vectalg::Multiindex& m)
{
  using namespace capd::vectalg;
  if(v.dimension()!=m.dimension())
    throw std::runtime_error("power(vector,multiindex) error: different dimensions of vector and multiindex");
  typename VectorType::ScalarType result=1;
  typename VectorType::const_iterator b=v.begin(), e=v.end();
  typename Multiindex::const_iterator i=m.begin();
  while(b!=e)
  {
    result *= power(*b,*i);
    ++b;
    ++i;
  }
  return result;
}

// -------------------------------------------------------------------------------

template<typename VectorType>
typename VectorType::ScalarType power(const VectorType& v, const capd::vectalg::Multipointer& m)
{
  using namespace capd::vectalg;
  typedef typename VectorType::ScalarType ScalarType;
  ScalarType result = capd::TypeTraits<ScalarType>::one();
  typename Multipointer::const_iterator b=m.begin(), e=m.end();
  while(b!=e)
  {
    typename Multipointer::const_iterator temp=b;
    int p = *b;
    do{
      ++b;
    }while(b!=e && *b==p);
    size_t n = b-temp;
    result *= power(v[p],(int)n);
  }
  return result;
}

#endif // _CAPD_VECTALG_MULTIINDEX_H_

/// @}

// --- End include/capd/vectalg/Multiindex.h ---

// --- Begin Multiindex.cpp ---

/////////////////////////////////////////////////////////////////////////////
/// @file vectalg/Multiindex.cpp
///
/// @author Daniel Wilczak
/////////////////////////////////////////////////////////////////////////////

// Copyright (C) 2000-2013 by the CAPD Group.
//
// This file constitutes a part of the CAPD library,
// distributed under the terms of the GNU General Public License.
// Consult  http://capd.ii.uj.edu.pl/ for details.

// #include "capd/basicalg/factrial.h"
// #include "capd/vectalg/Multiindex.h"
// #include "capd/vectalg/Container.hpp"

namespace capd{
namespace vectalg{

// static members
std::vector<Multipointer::IndicesSet> Multipointer::indices;
Multipointer::size_type Multipointer::maxKnownLevel=0;

// static functions

// ---------------------------------------------------------------------

const Multipointer::IndicesSet& Multipointer::generateList(size_type p, size_type k)
{
   if(k<0 || k>p || p<1)
      throw std::runtime_error("subIndices: wrong arguments! Call with 1<= second <= first");

   if(p>maxKnownLevel)
   {
      indices.resize(((p+1)*p)/2);
      for(size_type i=maxKnownLevel+1;i<=p;++i)
         computeNextLevel();
   }
   return getList(p,k);
}

// ---------------------------------------------------------------------

void Multipointer::computeNextLevel()
{
  if(maxKnownLevel==0)
  {
    Multipointer first(1);
    MultipointersVector mv;
    mv.push_back(first);
    indices[0].push_back(mv);
    maxKnownLevel=1;
  }else
  {
    size_type p = maxKnownLevel+1;
    Multipointer first(p,true);

    int i=0;
    iterator b=first.begin(), e=first.end();
    while(b!=e)
    {
       *b = i;
       ++b;
       ++i;
    }
    //for k=0
    MultipointersVector mv;
    mv.push_back(first);
    getList(p,1).push_back(mv);

    for(size_type k=2;k<=maxKnownLevel;++k)
    {

      Multipointer last(1,true);
      last[0] = maxKnownLevel;
      IndicesSet& current = getList(p,k);

      const IndicesSet& lower = getList(maxKnownLevel,k-1);
      IndicesSet::const_iterator b = lower.begin(), e=lower.end();
      while(b!=e)
      {
        MultipointersVector mv = *b;
        mv.push_back(last);
        current.push_back(mv);
        ++b;
      }

      const IndicesSet& higher = getList(maxKnownLevel,k);
      b = higher.begin();
      e = higher.end();
      while(b!=e)
      {
        for(unsigned int s=0;s<(*b).size();++s)
        {
          MultipointersVector copy = *b;
          copy[s] = sumMultipointers((*b)[s],last);
          current.push_back(copy);
        }
        ++b;
      }
    }

    // for k=maxKnownLevel+1
    MultipointersVector last;
    for(size_type s=0;s<p;++s)
    {
      Multipointer mv(1,true);
      mv[0] = s;
      last.push_back(mv);
    }
    getList(p,p).push_back(last);

     ++maxKnownLevel;
  }
}

// ---------------------------------------------------------------------

void Multiindex::generateList(size_type n, size_type k, IndicesSet& result)
{
   result.resize(k);
   //we storage the first level, i.e. k=1
   size_type i;
   for(i=0;i<n;++i){
     Multiindex mi(1,true);
     mi[0] = i;
     result[0].push_back(mi);
   }

   for(i=1;i<k;++i)
   {
      MultiindexVector::iterator b = result[i-1].begin(), e=result[i-1].end();
      while(b!=e)
      {
         for(size_type j=0;j<n;++j)
         {
            Multiindex m(i+1);
            for(size_type c=0;c<i;++c)
               m[c] = (*b)[c];
            m[i] = j;
            result[i].push_back(m);
         }
         ++b;
      }
   }
}

// ---------------------------------------------------------------------

bool Multipointer::hasNext(size_type dim){
  if(this->dimension())
  {
    iterator b=begin(), e=end(), e1=end();
    do
    {
      --e;
      if( ++(*e) % dim )
      {
        int p=*e;
        ++e;
        while(e!=e1)
        {
          *e=p;
          ++e;
        }
        return true;
      }
    }while(b!=e);
  }
  return false;
}

// ---------------------------------------------------------------------

long Multipointer::factorial() const{
   const_iterator b=begin(), e=end();
   long result=1;
   while(b!=e)
   {
      const_iterator temp=b;
      int p = *b;
      do{
         ++b;
      }while(b!=e && *b==p);
      size_t n = b-temp;
      if(n>1)
         result *= ::factorial(n);
   }
   return result;
}

// ---------------------------------------------------------------------
/// Returns multipointer containing entries which indices are in mp
///
/// e.g. for a = (1,3,3,6,7)  mp=(1,2,4)
///   a.subMultipointer(mp)  returns (3,3,7)
///
Multipointer Multipointer::subMultipointer(const Multipointer& mp) const
{
   Multipointer result(mp.dimension(),true);
   iterator i=result.begin();
   const_iterator j=begin();
   const_iterator b=mp.begin(), e=mp.end();
   while(b!=e)
   {
      (*i) = *(j+(*b));
      ++i;
      ++b;
   }
   return result;
}

// ---------------------------------------------------------------------
/// returns sum of the multiindex entries
Multiindex::size_type Multiindex::module() const{
   const_iterator b=begin(), e=end();
   int result=0;
   while(b!=e)
   {
      result += (*b); // assume Multiindex has nonnegative coordinates only
      ++b;
   }
   return result;
}

// ---------------------------------------------------------------------
/// for multiindex (a,b,..,n) returns a!b!...n!
long Multiindex::factorial() const{
   const_iterator b=begin(), e=end();
   long result=1;
   while(b!=e)
   {
      if((*b)>1)
         result *= ::factorial(*b);
      ++b;
   }
   return result;
}

// ---------------------------------------------------------------------

Multiindex::Multiindex(size_type dim, const Multipointer& mp) : Vector<int,0>(dim)
{
   Multipointer::const_iterator b=mp.begin(), e=mp.end();
   while(b!=e)
   {
      ++ ((*this))[*b];
      ++b;
   }
}


// ---------------------------------------------------------------------

Multipointer::Multipointer(const Multiindex& mi) : Vector<int,0>(mi.module(),true)
{
   iterator i=begin();
   for(size_type j=0;j<mi.dimension();++j)
   {
      for(int r=0;r<mi[j]; ++r)
      {
         (*i) = j;
         ++i;
      }
   }
}

// ---------------------------------------------------------------------

// the following function computes the next multipointer after mp
// it returns false if 'a' is zero multiindex

bool Multiindex::hasNext()
{
  if(this->dimension()<2) return false;
  if(this->data[0]!=0) {
    this->data[0]--;
    this->data[1]++;
    return true;
  }
  for(size_type i=1;i<this->dimension()-1;++i)
  {
    if(this->data[i]!=0){
      this->data[0] = this->data[i]-1;
      this->data[i]=0;
      this->data[i+1]++;
      return true;
    }
  }
  return false;
}

// ---------------------------------------------------------------------

bool Multiindex::hasNext(int* a, int* b) const {
  for(size_type i=0;i<this->dimension();++i){
    if(b[i]>0){
      b[i]--;
      a[i]++;
      return true;
    }
    b[i] = (*this)[i];
    a[i] = 0;
  }
  return false;
}

// ---------------------------------------------------------------------

bool Multiindex::hasNext(int* a, int* b, size_type j) const{
  for(size_type i=0;i<this->dimension();++i){
    if(b[i]> (i==j)){
      b[i]--;
      a[i]++;
      return true;
    }
    a[i] = (i==j);
    b[i] = (*this)[i]-a[i];
  }
  return false;
}

// ---------------------------------------------------------------------

// this function just concatenate sorted Multipointers to the another sorted Multipointer
Multipointer sumMultipointers(const Multipointer& x, const Multipointer& y)
{
   Multipointer result(x.module()+y.module());
   Multipointer::const_iterator xb=x.begin(), xe=x.end(), yb=y.begin(), ye=y.end();
   Multipointer::iterator b=result.begin();
   while(xb!=xe && yb!=ye)
   {
      if((*xb)<(*yb))
      {
         (*b) = (*xb);
         ++xb;
      }else{
         (*b) = (*yb);
         ++yb;
      }
      ++b;
   }
   if(xb==xe)
   {
      xb = yb;
      xe = ye;
   }
   while(xb!=xe)
   {
      (*b) = (*xb);
      ++b;
      ++xb;
   }
   return result;
}

Multipointer addIndex(const Multipointer & mp, int index) {
  Multipointer result(mp.dimension() + 1);
  Multipointer::iterator res = result.begin();
  Multipointer::const_iterator src = mp.begin(), end = mp.end();
  while(src != end){
    if((index>=0) && (index < *src)) {
      *res = index;
      index = -1;
      res++;
    }
    *res = *src;
    ++res; ++src;
  }
  if(res != result.end()){
    *res = index;
  }
  return result;
}

Multipointer push_back(const Multipointer & mp, int index) {
  Multipointer result(mp.dimension() + 1);
  Multipointer::iterator res = result.begin();
  Multipointer::const_iterator src = mp.begin(), end = mp.end();
  while(src != end){
    *res = *src;
    ++res; ++src;
  }
  *res = index;
  return result;
}

// ----------------------------------------------------------

inline int computeNewton(int d,int l)
{
  return binomial(d+l-1,l);
}

// ----------------------------------------------------------

// The following procedure computes an index of of an element in array that corresponds to the multipointer.
Multipointer::size_type Multipointer::index(size_type dim, size_type maxDegree) const
{
  size_type level = this->module(); // in fact dimension
  if (level<=0) return 0;
  if(level>maxDegree){
    throw std::range_error("Multipointer::index(int dim, int maxDegree): requested degree is to large");
  }
  size_type result=0,i;
  size_type prev = 0;

  for(i=0;i<level;++i)
  {
    if((*this)[i]-prev){
      result += (computeNewton(dim-prev,level-i) - computeNewton(dim-(*this)[i],level-i));
      prev = (*this)[i];
    }
  }
  return result;
}

// ----------------------------------------------------------

Multipointer::size_type Multipointer::index(size_type dim, size_type maxDegree, const Multipointer& sub) const
{
  size_type level = sub.module();
  if (level<=0) return 0;
  if(level>maxDegree)
    throw std::range_error("Multipointer::index(size_type dim, size_type maxDegree,Multipointer): requested degree is to large");

  size_type result=0,i;
  size_type prev = 0;

  for(i=0;i<level;++i)
  {
    size_type s = (*this)[sub[i]];
    if(s-prev){
      result += (computeNewton(dim-prev,level-i) - computeNewton(dim-s,level-i));
      prev = s;
    }
  }
  return result;
}

// ----------------------------------------------------------

Multiindex::size_type Multiindex::index(size_type maxDegree) const
{
  size_type level = this->module(); // sum norm
  if (level<=0) return 0;
  if(level>maxDegree)
    throw std::range_error("Multiindex::index(size_type maxDegree): requested degree is to large");

  size_type result=0,i, prev=0;
  for(i=0;i<this->dimension();++i)
  {
    if((*this)[i]!=0){
      result+= (computeNewton(this->dimension()-prev,level) - computeNewton(this->dimension()-i,level));
      prev = i;
      level -= (*this)[i];
    }
  }
  return result;
}

// has is not really important it does a bit less than indexCount, but the only time
// hasIndex will perform less operations than indexCount will be only when the indexes
// will actually be found and than we will still need to compute indexCount
bool hasIndex(const  Multipointer & mp, int index) {
  for (Multipointer::const_iterator it = mp.begin() ; it != mp.end() && *it <= index; ++it) {
    if (*it == index) return true;
  }
  return false;
}


int indexCount(const Multipointer &mp, int index) {

  int count = 0;
  for (Multipointer::const_iterator it = mp.begin() ; it != mp.end() && *it <= index; ++it) {
    count += (*it == index);
  }
  return count;
}

Multipointer removeIndex(const Multipointer & mp, int index) {
  Multipointer result(mp.dimension() -1 );
  Multipointer::iterator res = result.begin();
  Multipointer::const_iterator src = mp.begin(), end = mp.end();
  bool removed = false;
  while(src != end){
    if (!removed && index == *src) {
    removed = true;
    ++src;
  }
  else{
    *res = *src;
    ++res; ++src;
  }
  }
  return result;
}

}} // namespace capd::vectalg


// --- End Multiindex.cpp ---

// --- Begin include/capd/diffAlgebra/CnContainer.h ---
/// @addtogroup diffAlgebra
/// @{

/////////////////////////////////////////////////////////////////////////////
/// @file CnContainer.h
///
/// @author Daniel Wilczak
/////////////////////////////////////////////////////////////////////////////

// Copyright (C) 2000-2012 by the CAPD Group.
//
// This file constitutes a part of the CAPD library,
// distributed under the terms of the GNU General Public License.
// Consult  http://capd.ii.uj.edu.pl/ for details.

#include <stdexcept>
// #include "capd/basicalg/factrial.h"
// #include "capd/vectalg/Multiindex.h"
// #include "capd/vectalg/Container.h"

#ifndef _CAPD_DIFFALGEBRA_CNCONTAINER_H_
#define _CAPD_DIFFALGEBRA_CNCONTAINER_H_

namespace capd{
namespace diffAlgebra{

using capd::vectalg::__size_type;
using capd::vectalg::__difference_type;

/**
 * The class is used to store coefficients of a multivariate polynomial of degree D
 * \f$ f:R^N->R^M \f$
 * Coefficients themselves can be polynomials as well.
 * Usually Object = Scalar or an univariate polynomial
 *
 * The total number of coefficients is equal to \f$ M {N+D\choose D} \f$
*/

template<typename Object, __size_type M, __size_type N, __size_type D>
class CnContainer : public capd::vectalg::Container<Object,M*N*D!=0 ? M*Binomial<N+D,D>::value : 0>
{
public:
  typedef capd::vectalg::Container<Object,M*N*D!=0 ? M*Binomial<N+D,D>::value : 0> BaseContainer;
  typedef Object ObjectType;
  typedef Object* iterator;
  typedef const Object* const_iterator;
  typedef capd::vectalg::Multipointer Multipointer;
  typedef capd::vectalg::Multiindex Multiindex;
  typedef __size_type size_type;
  typedef __difference_type difference_type;

  CnContainer(size_type m, size_type n, size_type d, const Object& p); ///< creates a container for polynomial of n variables, m components and degree d. Each element will be set to p.
  CnContainer(size_type m, size_type n, size_type d);                  ///< creates a container for polynomial of n variables, m components and degree d. Default constructor will be used to initialize each element in the container.
  CnContainer& operator=(const Object& p);           ///< assigns object p to each element of the container

  CnContainer(const CnContainer& v) = default;
  CnContainer(CnContainer&& v) : BaseContainer(std::move(v)), m_N(v.m_N), m_M(v.m_M), m_D(v.m_D) {}
  CnContainer & operator=(const CnContainer & v) = default;
  CnContainer & operator=(CnContainer && v) {
    swap(*this, v);
    return *this;
  }

  size_type imageDimension() const; ///< returns number of polynomials (dimension of counterdomain)
  size_type dimension() const;      ///< returns number of variables of the polynomial
  size_type degree() const;         ///< returns degree of the polynomial

//  void resize(int newDegree, bool copyData = true);                   ///< changes degree of the polynomial
//  void resize(int newDegree, int newDimension, bool copyData = true); ///< changes degree and the number of variables of the polynomial
//  void resize(int newDegree, int newDimension, int newimageDim, bool copyData = true); ///< changes degree and the number of variables of the polynomial

// indexing

  using BaseContainer::operator[]; //< direct access to an element by its absolute position in the container

  Object& operator()(size_type i, const Multipointer& mp);  ///< selection of coefficient of i-th component that correspond to multipointer mp
  Object& operator()(size_type i, const Multipointer&, const Multipointer&);
  Object& operator()(size_type i, const Multiindex& mi); ///< selection of coefficient of i-th component that correspond to multiindex mi

  const Object& operator()(size_type i, const Multipointer&) const; ///< selection of coefficient of i-th component that correspond to multipointer mp
  const Object& operator()(size_type i, const Multipointer&, const Multipointer&) const;
  const Object& operator()(size_type i, const Multiindex&) const; ///< selection of coefficient of i-th component that correspond to multiindex mi

// operators for C^0, C^1, C^2 and C^3 algorithms
  Object& operator()(size_type i);                      ///< returns constant term of the i-th component of polynomial
  Object& operator()(size_type i, size_type j);               ///< returns reference to a coefficient in linear part, i.e. \f$ df_i/dx_j \f$
  Object& operator()(size_type i, size_type j, size_type c);        ///< returns reference to a coefficient in second order part, i.e. \f$ d^2f_i/dx_jdx_c \f$
  Object& operator()(size_type i, size_type j, size_type c, size_type k); ///< returns reference to a coefficient in third order part, i.e. \f$ d^3f_i/dx_jdx_cdx_k \f$

  const Object& operator()(size_type i) const;                      ///< returns constant term of the i-th component of polynomial
  const Object& operator()(size_type i, size_type j) const;               ///< returns read only reference to a coefficient in linear part, i.e. \f$ df_i/dx_j \f$
  const Object& operator()(size_type i, size_type j, size_type c) const;        ///< returns read only reference to a coefficient in second order part, i.e. \f$ d^2f_i/dx_jdx_c \f$
  const Object& operator()(size_type i, size_type j, size_type c, size_type k) const; ///< returns read only reference to a coefficient in third order part, i.e. \f$ d^3f_i/dx_jdx_cdx_k \f$

// iterators
  using BaseContainer::begin;   //< iterator selection. Returns iterator to the first element in container
  using BaseContainer::end;     //< iterator selection. Returns iterator to the first element in container
  using BaseContainer::clear;

  iterator begin(size_type i);        ///< iterator selection. Returns iterator to the first coefficient of the i-th component
  iterator end(size_type i);          ///< iterator selection. Returns iterator to an element after the last element the i-th component
  iterator begin(size_type i, size_type d); ///< iterator selection. Returns iterator to the first coefficient of the i-th component of the homogeneous part of degree 'd'
  iterator end(size_type i, size_type d);   ///< iterator selection. Returns iterator to an element after the last coefficient of the i-th component of the homogeneous part of degree 'd'

  const_iterator begin(size_type i) const;            ///< iterator selection. Returns iterator to the first coefficient of the i-th component
  const_iterator end(size_type i) const;              ///< iterator selection. Returns iterator to an element after the last element the i-th component
  const_iterator begin(size_type i, size_type d) const;     ///< iterator selection. Returns iterator to the first coefficient of the i-th component of the homogeneous part of degree 'd'
  const_iterator end(size_type i, size_type d) const;       ///< iterator selection. Returns iterator to an element after the last coefficient of the i-th component of the homogeneous part of degree 'd'

/**
 * Selection of elements by multipointers.
 *
 * Iterators do not give information about the index of partial derivative.
 * Access by multipointer is significantly slower than by iterator because the multipointer must be recomputed to the index in array.
 *
 * Typical usage of multipointers is as follows:
 * <code>
 * Multipointer mp = cnContainer.first(d);
 * int i = ...; // fix i-th component
 * do{
 *   // do something
 *   cout << mp << "\t" << cnContainer(i,mp) << endl;
 * }while(cnContainer.hasNext(mp));
 * </code>
 * Iterators and multipointers read coefficients of a homogeneous polynomial in the same order.
 */
  Multipointer first(size_type d) const;
  bool hasNext(Multipointer&) const; ///< see description of the method first.
  bool hasNext(Multiindex&) const; ///< see description of the method first.

  friend void swap(CnContainer & A, CnContainer & B){ ///< swaps the content of two containers
    std::swap(A.m_N, B.m_N);
    std::swap(A.m_M, B.m_M);
    std::swap(A.m_D, B.m_D);
    swap(static_cast<BaseContainer &>(A),static_cast<BaseContainer &>(B));
  }

protected:
  size_type m_N; ///< number of variables
  size_type m_M; ///< number of components
  size_type m_D; ///< total degree of polynomial
}; // the end of class CnContainer

// ------------------- member definitions -------------------
// the following function computes the next multipointer after mp
// it returns false if mp is the last multipointer on a given level

template<typename Object, __size_type M, __size_type N, __size_type D>
bool CnContainer<Object,M,N,D>::hasNext(Multipointer& mp) const
{
  return mp.hasNext(this->dimension());
}

template<typename Object, __size_type M, __size_type N, __size_type D>
bool CnContainer<Object,M,N,D>::hasNext(Multiindex& mp) const
{
  return mp.hasNext();
}

// ------------------- member definitions -------------------

template<typename Object, __size_type M, __size_type N, __size_type D>
CnContainer<Object,M,N,D>& CnContainer<Object,M,N,D>::operator=(const Object& p)
{
  iterator b=begin(), e=end();
  while(b!=e)
  {
    *b=p;
    ++b;
  }
  return *this;
}

// ----------------------------------------------------------

template<typename Object, __size_type M, __size_type N, __size_type D>
inline CnContainer<Object,M,N,D>::CnContainer(size_type m, size_type n, size_type d, const Object& p)
  : BaseContainer(m*binomial(n+d,d),true)
{
  m_N = N>0 ? N : n;
  m_M = M>0 ? M : m;
  m_D = D>0 ? D : d;

  iterator b = this->begin(), e=this->end();
  while(b!=e)
  {
    *b = p;
    ++b;
  }
}

// ----------------------------------------------------------
/*
template<typename Object, int M, int N, int D>
void CnContainer<Object,M,N,D>::resize(int newDegree,  bool copyData)
{
  if(newDegree == m_D)
    return;

  BaseContainer::resize(m_M*newton(m_N+m_D,m_D));
  if(copyData){
    int minDegree = m_D < newDegree ? m_D : newDegree;
    for(int i=0;i<m_N;++i)
    {
      iterator b = begin(i), e=end(i,minDegree);
      Object* p =  newData+ i*newton(m_dim+newRank,m_dim);
      while(b!=e)
      {
        *p = *b;
        ++p;
        ++b;
      }
    }
  }
  delete [] m_data;
  m_data = newData;
  m_rank = newRank;
  m_size = newSize;
}
*/
// ----------------------------------------------------------

/**
 * Resizes CnContainer
 * @param newRank       new maximal order
 * @param newDimension  new dimension
 * @param copyData      flag that controls if data is copied
 */
/*
template<typename Object, int M, int N, int D>
void CnContainer<Object,M,N,D>::resize(int newRank, int newDimension, bool copyData)
{
  if((newRank == m_rank) && (newDimension == m_dim))
    return;

  int newSize = newDimension*newton(newDimension+newRank,newDimension);

  Object* newData = new Object[newSize];
  if(copyData){
    int minRank = m_rank < newRank ? m_rank : newRank;
    int minDim = m_dim < newDimension ? m_dim : newDimension;
    for(int i=0; i< minDim; ++i){
      iterator b = begin(i),
          e = end(i,minRank);
      Object* p =  newData+ i*newton(newDimension+newRank,newDimension);
      while(b!=e)
      {
        *p = *b;
        ++p;
        ++b;
      }
    }
  }
  delete [] m_data;
  m_data = newData;
  m_rank = newRank;
  m_size = newSize;
  m_dim = newDimension;
}
*/
// ------------------- inline definitions -------------------

template<typename Object, __size_type M, __size_type N, __size_type D>
inline typename CnContainer<Object,M,N,D>::Multipointer CnContainer<Object,M,N,D>::first(size_type degree) const
{
  return Multipointer(degree);
}

// ----------------------------------------------------------

template<typename Object, __size_type M, __size_type N, __size_type D>
inline CnContainer<Object,M,N,D>::CnContainer(size_type m, size_type n, size_type d)
  : BaseContainer(m*binomial(n+d,d))
{
  m_N = N>0 ? N : n;
  m_M = M>0 ? M : m;
  m_D = D>0 ? D : d;
}

// ----------------------------------------------------------

template<typename Object, __size_type M, __size_type N, __size_type D>
inline typename CnContainer<Object,M,N,D>::size_type CnContainer<Object,M,N,D>::dimension() const
{
  return m_N;
}

// ----------------------------------------------------------

template<typename Object, __size_type M, __size_type N, __size_type D>
inline typename CnContainer<Object,M,N,D>::size_type CnContainer<Object,M,N,D>::imageDimension() const
{
  return m_M;
}

// ----------------------------------------------------------

template<typename Object, __size_type M, __size_type N, __size_type D>
inline typename CnContainer<Object,M,N,D>::size_type CnContainer<Object,M,N,D>::degree() const
{
  return m_D;
}

// ----------------------------------------------------------

template<typename Object, __size_type M, __size_type N, __size_type D>
inline Object& CnContainer<Object,M,N,D>::operator()(size_type i, const Multipointer& mp)
{
  return *(begin(i,mp.dimension()) + mp.index(m_N,m_D));
}

// ----------------------------------------------------------

template<typename Object, __size_type M, __size_type N, __size_type D>
inline const Object& CnContainer<Object,M,N,D>::operator()(size_type i, const Multipointer& mp) const
{
  return *(begin(i,mp.dimension()) + mp.index(m_N,m_D));
}

// ----------------------------------------------------------

template<typename Object, __size_type M, __size_type N, __size_type D>
inline Object& CnContainer<Object,M,N,D>::operator()(size_type i, const Multipointer& mp, const Multipointer& sub)
{
  return *(begin(i,sub.dimension()) + mp.index(m_N,m_D,sub));
}

// ----------------------------------------------------------

template<typename Object, __size_type M, __size_type N, __size_type D>
inline const Object& CnContainer<Object,M,N,D>::operator()(size_type i, const Multipointer& mp, const Multipointer& sub) const
{
  return *(begin(i,sub.dimension()) + mp.index(m_N,m_D,sub));
}

// ----------------------------------------------------------

template<typename Object, __size_type M, __size_type N, __size_type D>
inline Object& CnContainer<Object,M,N,D>::operator()(size_type i, const Multiindex& mi)
{
  if(this->dimension()!=mi.dimension())
    throw std::runtime_error("CnContainer::operator(int,Multiindex) - incompatible dimensions of CnContainer and Multiindex");
  return *(begin(i,mi.module()) + mi.index(m_D));
}

// ----------------------------------------------------------

template<typename Object, __size_type M, __size_type N, __size_type D>
inline const Object& CnContainer<Object,M,N,D>::operator()(size_type i, const Multiindex& mi) const
{
  if(this->dimension()!=mi.dimension())
    throw std::runtime_error("CnContainer::operator(int,Multiindex) - incompatible dimensions of CnContainer and Multiindex");
  return *(begin(i,mi.module()) + mi.index(m_D));
}

// -------------- indexing for C^0-C^3 algorithms ------------

template<typename Object, __size_type M, __size_type N, __size_type D>
inline Object& CnContainer<Object,M,N,D>::operator()(size_type i)
{
  return *(this->begin() + i*binomial(m_D+m_N,m_D));
}

// ----------------------------------------------------------

template<typename Object, __size_type M, __size_type N, __size_type D>
inline const Object& CnContainer<Object,M,N,D>::operator()(size_type i) const
{
  return *(this->begin() + i*binomial(m_D+m_N,m_D));
}

// ----------------------------------------------------------

template<typename Object, __size_type M, __size_type N, __size_type D>
inline Object& CnContainer<Object,M,N,D>::operator()(size_type i, size_type j)
{
  return *(begin(i,1)+j);
}

// ----------------------------------------------------------

template<typename Object, __size_type M, __size_type N, __size_type D>
inline const Object& CnContainer<Object,M,N,D>::operator()(size_type i, size_type j) const
{
  return *(begin(i,1)+j);
}

// ----------------------------------------------------------

template<typename Object, __size_type M, __size_type N, __size_type D>
inline Object& CnContainer<Object,M,N,D>::operator()(size_type i, size_type j, size_type c)
{
  return j<=c ?
    *(begin(i,2)+c-(j*(j+1))/2+j*m_N) :
    *(begin(i,2)+j-(c*(c+1))/2+c*m_N);
}

// ----------------------------------------------------------

template<typename Object, __size_type M, __size_type N, __size_type D>
inline const Object& CnContainer<Object,M,N,D>::operator()(size_type i, size_type j, size_type c) const
{
  return j<=c ?
    *(begin(i,2)+c-(j*(j+1))/2+j*m_N) :
    *(begin(i,2)+j-(c*(c+1))/2+c*m_N);
}

// ----------------------------------------------------------

template<typename Object, __size_type M, __size_type N, __size_type D>
inline Object& CnContainer<Object,M,N,D>::operator()(size_type i, size_type j, size_type c, size_type k)
{
  // assume j<=c<=k
  if(c<j || k<c)
    throw std::runtime_error("CnContainer::operator(int,int,int,int) - incorrect indexes");
  return *(
    begin(i,3) +
    (j*( (j-1)*(j-2) + 3*m_N*(m_N-j+2) ))/6    +   ((j-c)*(c+j-2*m_N-1))/2 + k-c
  );
}

// ----------------------------------------------------------

template<typename Object, __size_type M, __size_type N, __size_type D>
inline const Object& CnContainer<Object,M,N,D>::operator()(size_type i, size_type j, size_type c, size_type k) const
{
  // assume j<=c<=k
  if(c<j || k<c)
    throw std::runtime_error("CnContainer::operator(int,int,int,int) - incorrect indexes");
  return *(
    begin(i,3) +
    (j*( (j-1)*(j-2) + 3*m_N*(m_N-j+2) ))/6    +   ((j-c)*(c+j-2*m_N-1))/2 + k-c
  );
}

// ----------------------------------------------------------

template<typename Object, __size_type M, __size_type N, __size_type D>
inline typename CnContainer<Object,M,N,D>::iterator CnContainer<Object,M,N,D>::begin(size_type i)
{
  return iterator(this->begin()+i*binomial(m_N+m_D,m_D));
}

// ----------------------------------------------------------

template<typename Object, __size_type M, __size_type N, __size_type D>
inline typename CnContainer<Object,M,N,D>::iterator CnContainer<Object,M,N,D>::end(size_type i)
{
  return iterator(this->begin()+(i+1)*binomial(m_N+m_D,m_D));
}

// ----------------------------------------------------------

template<typename Object, __size_type M, __size_type N, __size_type D>
inline typename CnContainer<Object,M,N,D>::iterator CnContainer<Object,M,N,D>::begin(size_type i, size_type degree)
{
  return degree==0
         ? iterator(this->begin()+ i*binomial(m_N+m_D,m_D))
         : iterator(this->begin()+ i*binomial(m_N+m_D,m_D) + binomial(m_N+degree-1,m_N));
}

// ----------------------------------------------------------

template<typename Object, __size_type M, __size_type N, __size_type D>
inline typename CnContainer<Object,M,N,D>::iterator CnContainer<Object,M,N,D>::end(size_type i, size_type degree)
{
  return iterator(this->begin() + i*binomial(m_N+m_D,m_D) + binomial(m_N+degree,m_N));
}

// ----------------------------------------------------------

template<typename Object, __size_type M, __size_type N, __size_type D>
inline typename CnContainer<Object,M,N,D>::const_iterator CnContainer<Object,M,N,D>::begin(size_type i) const
{
  return const_iterator(this->begin()+i*binomial(m_N+m_D,m_N));
}

// ----------------------------------------------------------

template<typename Object, __size_type M, __size_type N, __size_type D>
inline typename CnContainer<Object,M,N,D>::const_iterator CnContainer<Object,M,N,D>::end(size_type i) const
{
  return const_iterator(this->begin()+(i+1)*binomial(m_N+m_D,m_D));
}

// ----------------------------------------------------------

template<typename Object, __size_type M, __size_type N, __size_type D>
inline typename CnContainer<Object,M,N,D>::const_iterator CnContainer<Object,M,N,D>::begin(size_type i, size_type degree) const
{
  return degree==0
         ? const_iterator(this->begin() + i*binomial(m_N+m_D,m_N))
         : const_iterator(this->begin() + i*binomial(m_N+m_D,m_N) + binomial(m_N+degree-1,m_N));
}

// ----------------------------------------------------------

template<typename Object, __size_type M, __size_type N, __size_type D>
inline typename CnContainer<Object,M,N,D>::const_iterator CnContainer<Object,M,N,D>::end(size_type i, size_type degree) const
{
  return const_iterator(this->begin() + i*binomial(m_N+m_D,m_N) + binomial(m_N+degree,m_N));
}

/// checks if two CnContainers are exactly the same.
template<typename Object, __size_type M, __size_type N, __size_type D>
bool operator == (const CnContainer<Object,M,N,D> & c1, const CnContainer<Object,M,N,D> & c2 ){
  if((c1.degree()()!=c2.degree()()) || (c1.dimension()!=c2.dimension()))
    return false;
  typename CnContainer<Object,M,N,D>::const_iterator it_c1 = c1.begin(), it_c2 = c2.begin(), end_c1 = c1.end();
  while(it_c1!=end_c1){
    if(*it_c1 != *it_c2)
       return false;
    it_c1++; it_c2++;
  }
  return true;
}

}} //namespace capd::diffAlgebra

#endif // _CAPD_DIFFALGEBRA_CNCONTAINER_H_

/// @}

// --- End include/capd/diffAlgebra/CnContainer.h ---

// --- Begin include/capd/autodiff/DagIndexer.h ---
/////////////////////////////////////////////////////////////////////////////
/// @file DagIndexer.h
///
/// @author Daniel Wilczak
/////////////////////////////////////////////////////////////////////////////

// Copyright (C) 2000-2017 by the CAPD Group.
//
// This file constitutes a part of the CAPD library,
// distributed under the terms of the GNU General Public License.
// Consult  http://capd.ii.uj.edu.pl/ for details.

#ifndef _CAPD_AUTODIFF_DAGINDEXER_H_
#define _CAPD_AUTODIFF_DAGINDEXER_H_

#include <vector>
// #include "capd/basicalg/TypeTraits.h"
// #include "capd/basicalg/factrial.h"
// #include "capd/vectalg/ColumnVector.h"
// #include "capd/diffAlgebra/CnContainer.h"

namespace capd{
namespace autodiff{

  inline int sumAndFindMax(const int a[], const int b[], int c[], const int n){
  int p=0,m=0;
  for(int i=0;i<n;++i)
  {
    c[i]=a[i]+b[i];
    if(c[i]>m){
      m=c[i];
      p=i;
    }
  }
  return p;
}

inline int findMax(const int c[], const int n){
  int p=0;
  for(int i=0;i<n;++i)
  {
    if(c[i]>c[p]){
      p=i;
    }
  }
  return p;
}

using capd::vectalg::__size_type;
using capd::vectalg::__difference_type;

// The code is written in almost pure C.
// Therefore there is a lot of integral arguments that can be easily incorrectly used.
// The following classes are wrappers for integral that assure
// correct order when passing arguments to functions.

// These classes should not be used in real computations, rather on development stage.
// Comment out the following line in order to use debug mode.
//#define _Dag_Indexer_Debug_Mode_

extern __size_type capd_c2jet_indices[21][20][20];
/*={
{},
{{2}},
{{3,4},{4,5}},
{{4,5,6},{5,7,8},{6,8,9}},
{{5,6,7,8},{6,9,10,11},{7,10,12,13},{8,11,13,14}},
{{6,7,8,9,10},{7,11,12,13,14},{8,12,15,16,17},{9,13,16,18,19},{10,14,17,19,20}},
{{7,8,9,10,11,12},{8,13,14,15,16,17},{9,14,18,19,20,21},{10,15,19,22,23,24},{11,16,20,23,25,26},{12,17,21,24,26,27}},
{{8,9,10,11,12,13,14},{9,15,16,17,18,19,20},{10,16,21,22,23,24,25},{11,17,22,26,27,28,29},{12,18,23,27,30,31,32},{13,19,24,28,31,33,34},{14,20,25,29,32,34,35}},
{{9,10,11,12,13,14,15,16},{10,17,18,19,20,21,22,23},{11,18,24,25,26,27,28,29},{12,19,25,30,31,32,33,34},{13,20,26,31,35,36,37,38},{14,21,27,32,36,39,40,41},{15,22,28,33,37,40,42,43},{16,23,29,34,38,41,43,44}},
{{10,11,12,13,14,15,16,17,18},{11,19,20,21,22,23,24,25,26},{12,20,27,28,29,30,31,32,33},{13,21,28,34,35,36,37,38,39},{14,22,29,35,40,41,42,43,44},{15,23,30,36,41,45,46,47,48},{16,24,31,37,42,46,49,50,51},{17,25,32,38,43,47,50,52,53},{18,26,33,39,44,48,51,53,54}},
{{11,12,13,14,15,16,17,18,19,20},{12,21,22,23,24,25,26,27,28,29},{13,22,30,31,32,33,34,35,36,37},{14,23,31,38,39,40,41,42,43,44},{15,24,32,39,45,46,47,48,49,50},{16,25,33,40,46,51,52,53,54,55},{17,26,34,41,47,52,56,57,58,59},{18,27,35,42,48,53,57,60,61,62},{19,28,36,43,49,54,58,61,63,64},{20,29,37,44,50,55,59,62,64,65}},
{{12,13,14,15,16,17,18,19,20,21,22},{13,23,24,25,26,27,28,29,30,31,32},{14,24,33,34,35,36,37,38,39,40,41},{15,25,34,42,43,44,45,46,47,48,49},{16,26,35,43,50,51,52,53,54,55,56},{17,27,36,44,51,57,58,59,60,61,62},{18,28,37,45,52,58,63,64,65,66,67},{19,29,38,46,53,59,64,68,69,70,71},{20,30,39,47,54,60,65,69,72,73,74},{21,31,40,48,55,61,66,70,73,75,76},{22,32,41,49,56,62,67,71,74,76,77}},
{{13,14,15,16,17,18,19,20,21,22,23,24},{14,25,26,27,28,29,30,31,32,33,34,35},{15,26,36,37,38,39,40,41,42,43,44,45},{16,27,37,46,47,48,49,50,51,52,53,54},{17,28,38,47,55,56,57,58,59,60,61,62},{18,29,39,48,56,63,64,65,66,67,68,69},{19,30,40,49,57,64,70,71,72,73,74,75},{20,31,41,50,58,65,71,76,77,78,79,80},{21,32,42,51,59,66,72,77,81,82,83,84},{22,33,43,52,60,67,73,78,82,85,86,87},{23,34,44,53,61,68,74,79,83,86,88,89},{24,35,45,54,62,69,75,80,84,87,89,90}},
{{14,15,16,17,18,19,20,21,22,23,24,25,26},{15,27,28,29,30,31,32,33,34,35,36,37,38},{16,28,39,40,41,42,43,44,45,46,47,48,49},{17,29,40,50,51,52,53,54,55,56,57,58,59},{18,30,41,51,60,61,62,63,64,65,66,67,68},{19,31,42,52,61,69,70,71,72,73,74,75,76},{20,32,43,53,62,70,77,78,79,80,81,82,83},{21,33,44,54,63,71,78,84,85,86,87,88,89},{22,34,45,55,64,72,79,85,90,91,92,93,94},{23,35,46,56,65,73,80,86,91,95,96,97,98},{24,36,47,57,66,74,81,87,92,96,99,100,101},{25,37,48,58,67,75,82,88,93,97,100,102,103},{26,38,49,59,68,76,83,89,94,98,101,103,104}},
{{15,16,17,18,19,20,21,22,23,24,25,26,27,28},{16,29,30,31,32,33,34,35,36,37,38,39,40,41},{17,30,42,43,44,45,46,47,48,49,50,51,52,53},{18,31,43,54,55,56,57,58,59,60,61,62,63,64},{19,32,44,55,65,66,67,68,69,70,71,72,73,74},{20,33,45,56,66,75,76,77,78,79,80,81,82,83},{21,34,46,57,67,76,84,85,86,87,88,89,90,91},{22,35,47,58,68,77,85,92,93,94,95,96,97,98},{23,36,48,59,69,78,86,93,99,100,101,102,103,104},{24,37,49,60,70,79,87,94,100,105,106,107,108,109},{25,38,50,61,71,80,88,95,101,106,110,111,112,113},{26,39,51,62,72,81,89,96,102,107,111,114,115,116},{27,40,52,63,73,82,90,97,103,108,112,115,117,118},{28,41,53,64,74,83,91,98,104,109,113,116,118,119}},
{{16,17,18,19,20,21,22,23,24,25,26,27,28,29,30},{17,31,32,33,34,35,36,37,38,39,40,41,42,43,44},{18,32,45,46,47,48,49,50,51,52,53,54,55,56,57},{19,33,46,58,59,60,61,62,63,64,65,66,67,68,69},{20,34,47,59,70,71,72,73,74,75,76,77,78,79,80},{21,35,48,60,71,81,82,83,84,85,86,87,88,89,90},{22,36,49,61,72,82,91,92,93,94,95,96,97,98,99},{23,37,50,62,73,83,92,100,101,102,103,104,105,106,107},{24,38,51,63,74,84,93,101,108,109,110,111,112,113,114},{25,39,52,64,75,85,94,102,109,115,116,117,118,119,120},{26,40,53,65,76,86,95,103,110,116,121,122,123,124,125},{27,41,54,66,77,87,96,104,111,117,122,126,127,128,129},{28,42,55,67,78,88,97,105,112,118,123,127,130,131,132},{29,43,56,68,79,89,98,106,113,119,124,128,131,133,134},{30,44,57,69,80,90,99,107,114,120,125,129,132,134,135}},
{{17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32},{18,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47},{19,34,48,49,50,51,52,53,54,55,56,57,58,59,60,61},{20,35,49,62,63,64,65,66,67,68,69,70,71,72,73,74},{21,36,50,63,75,76,77,78,79,80,81,82,83,84,85,86},{22,37,51,64,76,87,88,89,90,91,92,93,94,95,96,97},{23,38,52,65,77,88,98,99,100,101,102,103,104,105,106,107},{24,39,53,66,78,89,99,108,109,110,111,112,113,114,115,116},{25,40,54,67,79,90,100,109,117,118,119,120,121,122,123,124},{26,41,55,68,80,91,101,110,118,125,126,127,128,129,130,131},{27,42,56,69,81,92,102,111,119,126,132,133,134,135,136,137},{28,43,57,70,82,93,103,112,120,127,133,138,139,140,141,142},{29,44,58,71,83,94,104,113,121,128,134,139,143,144,145,146},{30,45,59,72,84,95,105,114,122,129,135,140,144,147,148,149},{31,46,60,73,85,96,106,115,123,130,136,141,145,148,150,151},{32,47,61,74,86,97,107,116,124,131,137,142,146,149,151,152}},
{{18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34},{19,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50},{20,36,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65},{21,37,52,66,67,68,69,70,71,72,73,74,75,76,77,78,79},{22,38,53,67,80,81,82,83,84,85,86,87,88,89,90,91,92},{23,39,54,68,81,93,94,95,96,97,98,99,100,101,102,103,104},{24,40,55,69,82,94,105,106,107,108,109,110,111,112,113,114,115},{25,41,56,70,83,95,106,116,117,118,119,120,121,122,123,124,125},{26,42,57,71,84,96,107,117,126,127,128,129,130,131,132,133,134},{27,43,58,72,85,97,108,118,127,135,136,137,138,139,140,141,142},{28,44,59,73,86,98,109,119,128,136,143,144,145,146,147,148,149},{29,45,60,74,87,99,110,120,129,137,144,150,151,152,153,154,155},{30,46,61,75,88,100,111,121,130,138,145,151,156,157,158,159,160},{31,47,62,76,89,101,112,122,131,139,146,152,157,161,162,163,164},{32,48,63,77,90,102,113,123,132,140,147,153,158,162,165,166,167},{33,49,64,78,91,103,114,124,133,141,148,154,159,163,166,168,169},{34,50,65,79,92,104,115,125,134,142,149,155,160,164,167,169,170}},
{{19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36},{20,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53},{21,38,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69},{22,39,55,70,71,72,73,74,75,76,77,78,79,80,81,82,83,84},{23,40,56,71,85,86,87,88,89,90,91,92,93,94,95,96,97,98},{24,41,57,72,86,99,100,101,102,103,104,105,106,107,108,109,110,111},{25,42,58,73,87,100,112,113,114,115,116,117,118,119,120,121,122,123},{26,43,59,74,88,101,113,124,125,126,127,128,129,130,131,132,133,134},{27,44,60,75,89,102,114,125,135,136,137,138,139,140,141,142,143,144},{28,45,61,76,90,103,115,126,136,145,146,147,148,149,150,151,152,153},{29,46,62,77,91,104,116,127,137,146,154,155,156,157,158,159,160,161},{30,47,63,78,92,105,117,128,138,147,155,162,163,164,165,166,167,168},{31,48,64,79,93,106,118,129,139,148,156,163,169,170,171,172,173,174},{32,49,65,80,94,107,119,130,140,149,157,164,170,175,176,177,178,179},{33,50,66,81,95,108,120,131,141,150,158,165,171,176,180,181,182,183},{34,51,67,82,96,109,121,132,142,151,159,166,172,177,181,184,185,186},{35,52,68,83,97,110,122,133,143,152,160,167,173,178,182,185,187,188},{36,53,69,84,98,111,123,134,144,153,161,168,174,179,183,186,188,189}},
{{20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38},{21,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56},{22,40,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73},{23,41,58,74,75,76,77,78,79,80,81,82,83,84,85,86,87,88,89},{24,42,59,75,90,91,92,93,94,95,96,97,98,99,100,101,102,103,104},{25,43,60,76,91,105,106,107,108,109,110,111,112,113,114,115,116,117,118},{26,44,61,77,92,106,119,120,121,122,123,124,125,126,127,128,129,130,131},{27,45,62,78,93,107,120,132,133,134,135,136,137,138,139,140,141,142,143},{28,46,63,79,94,108,121,133,144,145,146,147,148,149,150,151,152,153,154},{29,47,64,80,95,109,122,134,145,155,156,157,158,159,160,161,162,163,164},{30,48,65,81,96,110,123,135,146,156,165,166,167,168,169,170,171,172,173},{31,49,66,82,97,111,124,136,147,157,166,174,175,176,177,178,179,180,181},{32,50,67,83,98,112,125,137,148,158,167,175,182,183,184,185,186,187,188},{33,51,68,84,99,113,126,138,149,159,168,176,183,189,190,191,192,193,194},{34,52,69,85,100,114,127,139,150,160,169,177,184,190,195,196,197,198,199},{35,53,70,86,101,115,128,140,151,161,170,178,185,191,196,200,201,202,203},{36,54,71,87,102,116,129,141,152,162,171,179,186,192,197,201,204,205,206},{37,55,72,88,103,117,130,142,153,163,172,180,187,193,198,202,205,207,208},{38,56,73,89,104,118,131,143,154,164,173,181,188,194,199,203,206,208,209}},
{{21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40},{22,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59},{23,42,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77},{24,43,61,78,79,80,81,82,83,84,85,86,87,88,89,90,91,92,93,94},{25,44,62,79,95,96,97,98,99,100,101,102,103,104,105,106,107,108,109,110},{26,45,63,80,96,111,112,113,114,115,116,117,118,119,120,121,122,123,124,125},{27,46,64,81,97,112,126,127,128,129,130,131,132,133,134,135,136,137,138,139},{28,47,65,82,98,113,127,140,141,142,143,144,145,146,147,148,149,150,151,152},{29,48,66,83,99,114,128,141,153,154,155,156,157,158,159,160,161,162,163,164},{30,49,67,84,100,115,129,142,154,165,166,167,168,169,170,171,172,173,174,175},{31,50,68,85,101,116,130,143,155,166,176,177,178,179,180,181,182,183,184,185},{32,51,69,86,102,117,131,144,156,167,177,186,187,188,189,190,191,192,193,194},{33,52,70,87,103,118,132,145,157,168,178,187,195,196,197,198,199,200,201,202},{34,53,71,88,104,119,133,146,158,169,179,188,196,203,204,205,206,207,208,209},{35,54,72,89,105,120,134,147,159,170,180,189,197,204,210,211,212,213,214,215},{36,55,73,90,106,121,135,148,160,171,181,190,198,205,211,216,217,218,219,220},{37,56,74,91,107,122,136,149,161,172,182,191,199,206,212,217,221,222,223,224},{38,57,75,92,108,123,137,150,162,173,183,192,200,207,213,218,222,225,226,227},{39,58,76,93,109,124,138,151,163,174,184,193,201,208,214,219,223,226,228,229},{40,59,77,94,110,125,139,152,164,175,185,194,202,209,215,220,224,227,229,230}}};
*/

inline __size_type index(__size_type dim, __size_type j, __size_type c)
{
  if(dim<=10)
      return capd_c2jet_indices[dim][j][c];
  //  return 1+c + ((j+1)*(2*dim-j))/2;
  // Remark: c is an integer! Therefore we use bit shifts and compute equivalent (c+1)(c+2dim)/2
  return
      j<=c ? 1+c + ((j+1)*(dim*2-j))/2
           : 1+j + ((c+1)*(dim*2-c))/2;
}

inline __size_type index(__size_type dim, __size_type j, __size_type c, __size_type k)
{
  // assume j<=c<=k
  if(c<j || k<c)
    throw std::runtime_error("capd::autodiff::index(size_type,size_type,size_type) - indices are not ordered");
  return k - c +
      (
        (1+dim)*(2+dim) +
        (j*(j-1)*(j-2))/3 +
        j*dim*(dim-j+2) +
        (j-c)*(c+j-2*dim-1)
      ) /2;
}

#ifdef _Dag_Indexer_Debug_Mode_

struct JetSize
{
  explicit JetSize(__size_type i) : m_i(i) {}

  inline operator __size_type (void) { return m_i; }
  __size_type m_i;
};

struct Order
{
  explicit Order(__size_type i) : m_i(i) {}

  inline operator __size_type (void) { return m_i; }
  __size_type m_i;
};

struct VarNo
{
  explicit VarNo(__size_type i) : m_i(i) {}

  inline operator __size_type (void) { return m_i; }
  __size_type m_i;
};

struct CoeffNo
{
  explicit CoeffNo(__size_type i) : m_i(i) {}

  inline operator __size_type (void) { return m_i; }
  __size_type m_i;
};

struct DerNo
{
  explicit DerNo(__size_type i) : m_i(i) {}

  inline operator __size_type (void) { return m_i; }
  __size_type m_i;
};

template<class ScalarType>
inline ScalarType& getC0Coeff(ScalarType* data, VarNo varNo, JetSize jetSize, CoeffNo coeffNo)
{
  return data[varNo*jetSize+coeffNo];
}

template<class ScalarType>
inline ScalarType& getC1Coeff(ScalarType* data, VarNo varNo, DerNo derNo, JetSize jetSize, Order order, CoeffNo coeffNo)
{
  return data[varNo*jetSize + (derNo+1)*(order+1) + coeffNo];
}

template<class ScalarType>
inline ScalarType& getC2Coeff(ScalarType* data, unsigned dim, VarNo varNo, DerNo j, DerNo c, JetSize jetSize, Order order, CoeffNo coeffNo)
{
  return data[varNo*jetSize + index(dim,j,c)*(order+1) + coeffNo];
}

template<class ScalarType>
inline ScalarType& getC3Coeff(ScalarType* data, unsigned dim, VarNo varNo, DerNo i, DerNo j, DerNo c, JetSize jetSize, Order order, CoeffNo coeffNo)
{
  return data[varNo*jetSize + index(dim,i,j,c)*(order+1) + coeffNo];
}

#else

  typedef __size_type VarNo;
  typedef __size_type DerNo;
  typedef __size_type CoeffNo;
  typedef __size_type JetSize;
  typedef __size_type Order;
  typedef __size_type Degree;
  typedef __size_type Dim;

  #define getC0Coeff(data,varNo,jetSize,coeffNo) data[varNo*jetSize+coeffNo]
  #define getC1Coeff(data,varNo,derNo,jetSize,order,coeffNo) data[varNo*jetSize + (derNo+1)*(order+1) + coeffNo]
  #define getC2Coeff(data,dim,varNo,j,c,jetSize,order,coeffNo) data[varNo*jetSize + index(dim,j,c)*(order+1) + coeffNo]
  #define getC3Coeff(data,dim,varNo,j,c,k,jetSize,order,coeffNo) data[varNo*jetSize + index(dim,j,c,k)*(order+1) + coeffNo]

#endif

/// Stores information about decomposition of a Multiinex 'z' into possible sums of x+y=z
/// Used to optimizs convolutions.
/// All the data here is redundant and precomputed to avoid extra runtime computation.
struct MultiindexData{
  typedef __size_type size_type;
  MultiindexData() : p(0),k(1), index(0) {}

  MultiindexData(capd::vectalg::Multiindex k, size_type order) :k(k){
    const size_type dim = k.dimension();
    const size_type deg = k.module();
    this-> p = findMax(k.begin(),dim);
    this->index = totalIndex(k,order+1,dim,deg);
    this->convolution.resize(order+1);
    this->convolutionFromEpToK.resize(order+1);
    capd::vectalg::Multiindex a(dim), b=k;
    if(this->index==0){
      for(unsigned coeffNo=0;coeffNo<=order;++coeffNo)
        for(unsigned j=0;j<=coeffNo;++j)
          convolution[coeffNo].push_back(IndexPair(j,coeffNo-j));
    } else {
      do{
        size_type ia = totalIndex(a,order+1,dim,deg);
        size_type ib = totalIndex(b,order+1,dim,deg);
        for(unsigned coeffNo=0;coeffNo<=order;++coeffNo)
          for(unsigned j=0;j<=coeffNo;++j){
            convolution[coeffNo].push_back(IndexPair(ia+j,ib+coeffNo-j));
            if(a[p]>0)
             convolutionFromEpToK[coeffNo].push_back(IndexPair(ia+j,ib+coeffNo-j));
          }
      }while(k.hasNext(a.begin(),b.begin()));
    }
  }

  static size_type totalIndex(const capd::vectalg::Multiindex& a, size_type order, size_type dim, size_type deg){
    size_type ma = a.module();
    return order*(a.index(deg) + (ma>0 ? binomial(dim+ma-1,dim) : 0));
  }
  size_type p; /// largest index in multiindex
  capd::vectalg::Multiindex k;
  size_type index; /// redundant data - index of k
  typedef std::pair<size_type,size_type> IndexPair;
  typedef std::vector< IndexPair > ConvolutionPairs;
  std::vector< ConvolutionPairs > convolution;
  std::vector< ConvolutionPairs > convolutionFromEpToK;

  const ConvolutionPairs& getConvolutionPairs(size_type coeffNo) const { return convolution[coeffNo]; }
  const ConvolutionPairs& getConvolutionPairsFromEpToK(size_type coeffNo) const { return convolutionFromEpToK[coeffNo]; }
};

template<class ScalarT>
class DagIndexer
{
public:

  typedef ScalarT ScalarType;
  typedef __size_type size_type;
  typedef capd::vectalg::ColumnVector<ScalarType,0> RefVectorType;
  typedef ScalarType* iterator;
  typedef const ScalarType* const_iterator;
  typedef capd::diffAlgebra::CnContainer<MultiindexData,0,0,0> IndexArray;

  DagIndexer(Dim domain=1, Dim image=1, Degree degree=1, size_type nodes=1, Order order=0);
  DagIndexer(const DagIndexer& dag);
  ~DagIndexer();

  DagIndexer& operator=(const DagIndexer& dag);
/*
  template<class iterator>
  inline void setVector(iterator b, iterator e)
  {
    ScalarType *c = m_coefficients;
    while(b!=e)
    {
      *c = *b;
      ++b;
      c += jetSize();
    }
  }
*/

  ScalarType& operator()(VarNo varNo,CoeffNo coeffNo) {  return getC0Coeff(m_coefficients,varNo,JetSize(m_timeJetSize),coeffNo); }
  ScalarType& operator()(VarNo varNo, DerNo derNo, CoeffNo coeffNo) { return getC1Coeff(m_coefficients,varNo,derNo,JetSize(m_timeJetSize),m_order,coeffNo); }
  ScalarType& operator()(VarNo varNo, DerNo j, DerNo c, CoeffNo coeffNo) { return getC2Coeff(m_coefficients,m_domainDimension,varNo,j,c,JetSize(m_timeJetSize),m_order,coeffNo); }
  ScalarType& operator()(VarNo varNo, DerNo i, DerNo j, DerNo c, CoeffNo coeffNo) { return getC3Coeff(m_coefficients,m_domainDimension,varNo,i,j,c,JetSize(m_timeJetSize),m_order,coeffNo); }

  Dim domainDimension() const { return m_domainDimension; }
  Dim imageDimension()  const { return m_imageDimension; }
  Dim degree()          const { return m_degree; }
  JetSize jetSize()          const { return JetSize(binomial(m_domainDimension+m_degree,m_degree)); }
  JetSize timeJetSize()      const { return JetSize(m_timeJetSize); }

  /**
   * This method defines a mask for computation of partial derivatives of the function represented by the instance.
   * Each element of the range [b,e) should be a valid Multiindex. The user can specify which partial derivatives he/she needs tp compute.
   * Dependent derivatives are added to the list automatically and those independent are not evaluated which significantly speeds up the computation.
   *
   * Example:
   * setMask({Multiindex({1,1,0}),Multiindex({2,0,0})});
   *
   * Here we request derivatives dx1dx2 and d^2x1. They depend on first order derivatives dx1 and dx2 which will be added automatically.
   *
   * @param [b,e) - iterator range of Multiindxes
   */
  template<class Iterator>
  void setMask(Iterator b, Iterator e);
  const bool* getMask() const { return m_mask; }
  bool getMask(size_type j) const { return getC1Coeff(m_mask,VarNo(0),DerNo(j),JetSize(m_timeJetSize),m_order,CoeffNo(0)); }
  bool getMask(size_type j, size_type c) const { return getC2Coeff(m_mask,m_domainDimension,VarNo(0),DerNo(j),DerNo(c),JetSize(m_timeJetSize),m_order,CoeffNo(0)); }
  void addMultiindexToMask(const capd::vectalg::Multiindex& i);
  void resetMask();

  ScalarType* coefficients()             { return m_coefficients;}
  const ScalarType* coefficients() const { return m_coefficients;}
  Order getOrder()              const { return m_order; }
  void setOrder(Order order);
  void resize(Dim domain, Dim image, Degree degree, size_type nodes, Order order);
  size_type numberOfNodes() const {return m_numberOfNodes;} ///< returns total number of nodes in DAG representing expression
  iterator begin();               ///< iterator selection. Returns iterator to the first element in container
  iterator end();                 ///< iterator selection. Returns iterator to the first element in container
  iterator begin(size_type i);          ///< iterator selection. Returns iterator to the first coefficient of the i-th component
  iterator end(size_type i);            ///< iterator selection. Returns iterator to an element after the last element the i-th component
  iterator begin(size_type i, size_type d);   ///< iterator selection. Returns iterator to the first coefficient of the i-th component of the homogeneous part of degree 'd'
  iterator end(size_type i, size_type d);     ///< iterator selection. Returns iterator to an element after the last coefficient of the i-th component of the homogeneous part of degree 'd'

  const_iterator begin() const;                 ///< iterator selection. Returns iterator to the first element in container
  const_iterator end() const;                   ///< iterator selection. Returns iterator to the first element in container
  const_iterator begin(size_type i) const;            ///< iterator selection. Returns iterator to the first coefficient of the i-th component
  const_iterator end(size_type i) const;              ///< iterator selection. Returns iterator to an element after the last element the i-th component
  const_iterator begin(size_type i, size_type d) const;     ///< iterator selection. Returns iterator to the first coefficient of the i-th component of the homogeneous part of degree 'd'
  const_iterator end(size_type i, size_type d) const;       ///< iterator selection. Returns iterator to an element after the last coefficient of the i-th component of the homogeneous part of degree 'd'

  const IndexArray& getIndexArray() const{
    return this->m_indexArray;
  }
private:
  void add(const capd::vectalg::Multiindex& i);
  void fillByZeroes();

  /// allocates memory when all parameters are known. All coefficients are set to zero.
  void allocate(Dim domain, Dim image, Degree degree, size_type nodes, Order order);

  /// allocates memory and copies data from an existing object.
  void allocateAndCopy(const DagIndexer& dag);

  /// precomputes arrays of indices for convolutions
  void createIndexArray();

  ScalarType* m_coefficients; ///< pointer to allocated memory
  Dim m_domainDimension;      ///< total dimension of the domain (with time, parameters, etc)
  Dim m_imageDimension;       ///< dimension of the counterdomain
  Degree m_degree;            ///< degree of jet (polynomial of space variables)

  size_type m_numberOfNodes;  ///< total number of nodes in DAG
  Order m_order;              ///< order of polynomial with respect to distinguished time variable

  JetSize m_timeJetSize;      ///< size of chunk of memory needed for (m_order+1) jets.
  IndexArray m_indexArray;    ///< array of precomputed pairs of indexes for computation of convolutions
  bool* m_mask;               ///< a pointer to mask of derivatives
};

// ------------------- iterator selections --------------------------

template<class T>
inline typename DagIndexer<T>::iterator DagIndexer<T>::begin(){
  return this->m_coefficients;
}

template<class T>
inline typename DagIndexer<T>::iterator DagIndexer<T>::end(){
  return this->m_coefficients + this->m_numberOfNodes*this->timeJetSize();
}

template<class T>
inline typename DagIndexer<T>::iterator DagIndexer<T>::begin(size_type i){
  return this->m_coefficients + i*this->timeJetSize();
}

template<class T>
inline typename DagIndexer<T>::iterator DagIndexer<T>::end(size_type i){
  return this->m_coefficients + (i+1)*this->timeJetSize();
}

template<class T>
inline typename DagIndexer<T>::iterator DagIndexer<T>::begin(size_type i, size_type d){
  return d==0
         ? this->begin(i)
         : this->begin(i) + (this->getOrder()+1)*binomial(this->m_domainDimension+d-1,d-1);
}

template<class T>
inline typename DagIndexer<T>::iterator DagIndexer<T>::end(size_type i, size_type d){
  return this->begin(i) + (this->getOrder()+1)*binomial(this->m_domainDimension+d,d);
}

// ------------------- const_iterator selections --------------------------

template<class T>
inline typename DagIndexer<T>::const_iterator DagIndexer<T>::begin() const{
  return this->m_coefficients;
}

template<class T>
inline typename DagIndexer<T>::const_iterator DagIndexer<T>::end() const {
  return this->m_coefficients + this->m_numberOfNodes*this->timeJetSize();
}

template<class T>
inline typename DagIndexer<T>::const_iterator DagIndexer<T>::begin(size_type i) const{
  return this->m_coefficients + i*this->timeJetSize();
}

template<class T>
inline typename DagIndexer<T>::const_iterator DagIndexer<T>::end(size_type i) const{
  return this->m_coefficients + (i+1)*this->timeJetSize();
}

template<class T>
inline typename DagIndexer<T>::const_iterator DagIndexer<T>::begin(size_type i, size_type d) const{
  return d==0
         ? this->begin(i)
         : this->begin(i) + (this->getOrder()+1)*binomial(this->m_domainDimension+d-1,d-1);
}

template<class T>
inline typename DagIndexer<T>::const_iterator DagIndexer<T>::end(size_type i, size_type d) const{
  return this->begin(i) + (this->getOrder()+1)*binomial(this->m_domainDimension+d,d);
}

template<class T>
template<class Iterator>
void DagIndexer<T>::setMask(Iterator b, Iterator e){
  if(this->m_mask)
    delete[] m_mask;
  this->m_mask = new bool[this->m_timeJetSize];
  std::fill(this->m_mask,this->m_mask+this->m_order+1,true);
  std::fill(this->m_mask+this->m_order+1,this->m_mask+this->m_timeJetSize,false);

  while(b!=e){
    this->add(*b);
    ++b;
  }
  this->fillByZeroes();
}

}} // namespace capd::autodiff

#endif

// --- End include/capd/autodiff/DagIndexer.h ---

// --- Begin include/capd/autodiff/DagIndexer.hpp ---
/////////////////////////////////////////////////////////////////////////////
/// @file DagIndexer.hpp
///
/// @author Daniel Wilczak
/////////////////////////////////////////////////////////////////////////////

// Copyright (C) 2000-2017 by the CAPD Group.
//
// This file constitutes a part of the CAPD library,
// distributed under the terms of the GNU General Public License.
// Consult  http://capd.ii.uj.edu.pl/ for details.

#ifndef _CAPD_AUTODIFF_DAGINDEXER_HPP_
#define _CAPD_AUTODIFF_DAGINDEXER_HPP_

#include <algorithm>
// #include "capd/autodiff/DagIndexer.h"
// #include "capd/vectalg/Container.hpp"

namespace capd{
namespace autodiff{

template<class Scalar>
DagIndexer<Scalar>::DagIndexer(Dim domain, Dim image, Degree degree, size_type nodes, Order order)
  : m_order(order), m_indexArray(1,domain,degree,MultiindexData()), m_mask(0)
{
  this->allocate(domain,image,degree,nodes,order);
}

// -----------------------------------------------------------------------------

template<class Scalar>
DagIndexer<Scalar>::DagIndexer(const DagIndexer& dag)
  : m_order(dag.m_order), m_indexArray(dag.m_indexArray), m_mask(0)
{
  this->allocateAndCopy(dag);
}

// -----------------------------------------------------------------------------

template<class Scalar>
DagIndexer<Scalar>& DagIndexer<Scalar>::operator=(const DagIndexer& dag)
{
  if(this == &dag) return *this;
  delete[] m_coefficients;
  resetMask();
  this->allocateAndCopy(dag);
  return *this;
}

// -----------------------------------------------------------------------------

template<class Scalar>
DagIndexer<Scalar>::~DagIndexer()
{
  delete[] m_coefficients;
  if(this->m_mask)
    delete[] m_mask;
}

// -----------------------------------------------------------------------------

template<class Scalar>
void DagIndexer<Scalar>::setOrder(Order order)
{
  size_type newTimeJetSize = (order+1)*this->jetSize();
  size_type blockSize = newTimeJetSize*m_numberOfNodes;
  ScalarType* coeff = new ScalarType[blockSize];
  std::fill(coeff,coeff+blockSize,TypeTraits<ScalarType>::zero());

  for(size_type i=0;i<this->m_numberOfNodes;++i)
    getC0Coeff(coeff,VarNo(i),JetSize(newTimeJetSize),CoeffNo(0)) = getC0Coeff(m_coefficients,VarNo(i),JetSize(m_timeJetSize),CoeffNo(0));
  delete[] m_coefficients;
  m_coefficients = coeff;

  if(m_mask){
    bool* new_mask = new bool[newTimeJetSize];
    for(size_type i=0;i<this->jetSize();++i){
      for(size_type j=0;j<=order;++j)
        new_mask[i*(order+1)+j] = m_mask[i*(m_order+1)];
    }
    delete[] m_mask;
    m_mask = new_mask;
  }

  this->m_order = Order(order);
  this->m_timeJetSize = newTimeJetSize;
  this->createIndexArray();
}

// -----------------------------------------------------------------------------

template<class Scalar>
void DagIndexer<Scalar>::resize(Dim domain, Dim image, Degree degree, size_type nodes, Order order)
{
  delete[] m_coefficients;
  resetMask();
  this->allocate(domain,image,degree,nodes,order);
}

// -----------------------------------------------------------------------------

template<class Scalar>
void DagIndexer<Scalar>::createIndexArray(){
  m_indexArray = IndexArray(1,m_domainDimension,m_degree,MultiindexData());
  m_indexArray[0] = MultiindexData(capd::vectalg::Multiindex(m_domainDimension),m_order);
  for(size_type i=1;i<=m_degree;++i)
  {
    capd::vectalg::Multipointer a = m_indexArray.first(i);
    do{
      capd::vectalg::Multiindex mi(m_domainDimension,a);
      m_indexArray(0,a) = MultiindexData(mi,m_order);
    }while(m_indexArray.hasNext(a));
  }
}

// -----------------------------------------------------------------------------

template<class Scalar>
void DagIndexer<Scalar>::allocate(Dim domain, Dim image, Degree degree, size_type nodes, Order order)
{
  this->m_domainDimension = domain;
  this->m_imageDimension = image;
  this->m_degree = degree;
  this->m_numberOfNodes = nodes;
  this->m_order = Order(order);
  this->m_timeJetSize = (this->m_order+1)*this->jetSize();
  size_type blockSize = this->m_numberOfNodes*this->m_timeJetSize;
  this->m_coefficients = new ScalarType[blockSize];
  std::fill(this->m_coefficients,this->m_coefficients+blockSize,TypeTraits<ScalarType>::zero());
  this->createIndexArray();
}

// -----------------------------------------------------------------------------

template<class Scalar>
void DagIndexer<Scalar>::allocateAndCopy(const DagIndexer& dag)
{
  this->m_domainDimension = dag.m_domainDimension;
  this->m_imageDimension = dag.m_imageDimension;
  this->m_degree = dag.m_degree;
  this->m_numberOfNodes = dag.m_numberOfNodes;
  this->m_order = dag.m_order;
  this->m_timeJetSize = dag.m_timeJetSize;
  this->m_indexArray = dag.m_indexArray;

  size_type blockSize = this->m_numberOfNodes*this->m_timeJetSize;
  this->m_coefficients = new ScalarType[blockSize];
  std::copy(dag.coefficients(),dag.coefficients()+blockSize,this->m_coefficients);

  if(dag.m_mask){
    this->m_mask = new bool[this->m_timeJetSize];
    std::copy(dag.m_mask,dag.m_mask+dag.m_timeJetSize,this->m_mask);
  }
}

template<class Scalar>
void DagIndexer<Scalar>::addMultiindexToMask(const capd::vectalg::Multiindex& mi){
  this->add(mi);
  this->fillByZeroes();
}

template<class Scalar>
void DagIndexer<Scalar>::add(const capd::vectalg::Multiindex& mi){
  const MultiindexData& m = this->getIndexArray()(0,mi);
  const MultiindexData::ConvolutionPairs& p = m.getConvolutionPairs(this->m_order);
  for(MultiindexData::ConvolutionPairs::const_iterator i=p.begin();i!=p.end();++i){
    this->m_mask[i->first] = true;
    this->m_mask[i->second] = true;
  }
}

template<class Scalar>
void DagIndexer<Scalar>::fillByZeroes(){
  for(size_type i=0;i<this->m_numberOfNodes;++i)
    std::fill(this->m_coefficients+i*this->timeJetSize()+1,this->m_coefficients+(i+1)*this->timeJetSize(),TypeTraits<ScalarType>::zero());
}

template<class Scalar>
void DagIndexer<Scalar>::resetMask(){
  if(m_mask){
    delete[] m_mask;
    m_mask = 0;
  }
}

}}

#endif

// --- End include/capd/autodiff/DagIndexer.hpp ---

// --- Begin DagIndexer.cpp ---
/*
 * DagIndexer.cpp
 *
 *  Created on: Aug 25, 2012
 *      Author: kapela
 */


// #include "capd/autodiff/DagIndexer.hpp"

namespace capd{
namespace autodiff{

__size_type capd_c2jet_indices[21][20][20]={
{},
{{2}},
{{3,4},{4,5}},
{{4,5,6},{5,7,8},{6,8,9}},
{{5,6,7,8},{6,9,10,11},{7,10,12,13},{8,11,13,14}},
{{6,7,8,9,10},{7,11,12,13,14},{8,12,15,16,17},{9,13,16,18,19},{10,14,17,19,20}},
{{7,8,9,10,11,12},{8,13,14,15,16,17},{9,14,18,19,20,21},{10,15,19,22,23,24},{11,16,20,23,25,26},{12,17,21,24,26,27}},
{{8,9,10,11,12,13,14},{9,15,16,17,18,19,20},{10,16,21,22,23,24,25},{11,17,22,26,27,28,29},{12,18,23,27,30,31,32},{13,19,24,28,31,33,34},{14,20,25,29,32,34,35}},
{{9,10,11,12,13,14,15,16},{10,17,18,19,20,21,22,23},{11,18,24,25,26,27,28,29},{12,19,25,30,31,32,33,34},{13,20,26,31,35,36,37,38},{14,21,27,32,36,39,40,41},{15,22,28,33,37,40,42,43},{16,23,29,34,38,41,43,44}},
{{10,11,12,13,14,15,16,17,18},{11,19,20,21,22,23,24,25,26},{12,20,27,28,29,30,31,32,33},{13,21,28,34,35,36,37,38,39},{14,22,29,35,40,41,42,43,44},{15,23,30,36,41,45,46,47,48},{16,24,31,37,42,46,49,50,51},{17,25,32,38,43,47,50,52,53},{18,26,33,39,44,48,51,53,54}},
{{11,12,13,14,15,16,17,18,19,20},{12,21,22,23,24,25,26,27,28,29},{13,22,30,31,32,33,34,35,36,37},{14,23,31,38,39,40,41,42,43,44},{15,24,32,39,45,46,47,48,49,50},{16,25,33,40,46,51,52,53,54,55},{17,26,34,41,47,52,56,57,58,59},{18,27,35,42,48,53,57,60,61,62},{19,28,36,43,49,54,58,61,63,64},{20,29,37,44,50,55,59,62,64,65}},
{{12,13,14,15,16,17,18,19,20,21,22},{13,23,24,25,26,27,28,29,30,31,32},{14,24,33,34,35,36,37,38,39,40,41},{15,25,34,42,43,44,45,46,47,48,49},{16,26,35,43,50,51,52,53,54,55,56},{17,27,36,44,51,57,58,59,60,61,62},{18,28,37,45,52,58,63,64,65,66,67},{19,29,38,46,53,59,64,68,69,70,71},{20,30,39,47,54,60,65,69,72,73,74},{21,31,40,48,55,61,66,70,73,75,76},{22,32,41,49,56,62,67,71,74,76,77}},
{{13,14,15,16,17,18,19,20,21,22,23,24},{14,25,26,27,28,29,30,31,32,33,34,35},{15,26,36,37,38,39,40,41,42,43,44,45},{16,27,37,46,47,48,49,50,51,52,53,54},{17,28,38,47,55,56,57,58,59,60,61,62},{18,29,39,48,56,63,64,65,66,67,68,69},{19,30,40,49,57,64,70,71,72,73,74,75},{20,31,41,50,58,65,71,76,77,78,79,80},{21,32,42,51,59,66,72,77,81,82,83,84},{22,33,43,52,60,67,73,78,82,85,86,87},{23,34,44,53,61,68,74,79,83,86,88,89},{24,35,45,54,62,69,75,80,84,87,89,90}},
{{14,15,16,17,18,19,20,21,22,23,24,25,26},{15,27,28,29,30,31,32,33,34,35,36,37,38},{16,28,39,40,41,42,43,44,45,46,47,48,49},{17,29,40,50,51,52,53,54,55,56,57,58,59},{18,30,41,51,60,61,62,63,64,65,66,67,68},{19,31,42,52,61,69,70,71,72,73,74,75,76},{20,32,43,53,62,70,77,78,79,80,81,82,83},{21,33,44,54,63,71,78,84,85,86,87,88,89},{22,34,45,55,64,72,79,85,90,91,92,93,94},{23,35,46,56,65,73,80,86,91,95,96,97,98},{24,36,47,57,66,74,81,87,92,96,99,100,101},{25,37,48,58,67,75,82,88,93,97,100,102,103},{26,38,49,59,68,76,83,89,94,98,101,103,104}},
{{15,16,17,18,19,20,21,22,23,24,25,26,27,28},{16,29,30,31,32,33,34,35,36,37,38,39,40,41},{17,30,42,43,44,45,46,47,48,49,50,51,52,53},{18,31,43,54,55,56,57,58,59,60,61,62,63,64},{19,32,44,55,65,66,67,68,69,70,71,72,73,74},{20,33,45,56,66,75,76,77,78,79,80,81,82,83},{21,34,46,57,67,76,84,85,86,87,88,89,90,91},{22,35,47,58,68,77,85,92,93,94,95,96,97,98},{23,36,48,59,69,78,86,93,99,100,101,102,103,104},{24,37,49,60,70,79,87,94,100,105,106,107,108,109},{25,38,50,61,71,80,88,95,101,106,110,111,112,113},{26,39,51,62,72,81,89,96,102,107,111,114,115,116},{27,40,52,63,73,82,90,97,103,108,112,115,117,118},{28,41,53,64,74,83,91,98,104,109,113,116,118,119}},
{{16,17,18,19,20,21,22,23,24,25,26,27,28,29,30},{17,31,32,33,34,35,36,37,38,39,40,41,42,43,44},{18,32,45,46,47,48,49,50,51,52,53,54,55,56,57},{19,33,46,58,59,60,61,62,63,64,65,66,67,68,69},{20,34,47,59,70,71,72,73,74,75,76,77,78,79,80},{21,35,48,60,71,81,82,83,84,85,86,87,88,89,90},{22,36,49,61,72,82,91,92,93,94,95,96,97,98,99},{23,37,50,62,73,83,92,100,101,102,103,104,105,106,107},{24,38,51,63,74,84,93,101,108,109,110,111,112,113,114},{25,39,52,64,75,85,94,102,109,115,116,117,118,119,120},{26,40,53,65,76,86,95,103,110,116,121,122,123,124,125},{27,41,54,66,77,87,96,104,111,117,122,126,127,128,129},{28,42,55,67,78,88,97,105,112,118,123,127,130,131,132},{29,43,56,68,79,89,98,106,113,119,124,128,131,133,134},{30,44,57,69,80,90,99,107,114,120,125,129,132,134,135}},
{{17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32},{18,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47},{19,34,48,49,50,51,52,53,54,55,56,57,58,59,60,61},{20,35,49,62,63,64,65,66,67,68,69,70,71,72,73,74},{21,36,50,63,75,76,77,78,79,80,81,82,83,84,85,86},{22,37,51,64,76,87,88,89,90,91,92,93,94,95,96,97},{23,38,52,65,77,88,98,99,100,101,102,103,104,105,106,107},{24,39,53,66,78,89,99,108,109,110,111,112,113,114,115,116},{25,40,54,67,79,90,100,109,117,118,119,120,121,122,123,124},{26,41,55,68,80,91,101,110,118,125,126,127,128,129,130,131},{27,42,56,69,81,92,102,111,119,126,132,133,134,135,136,137},{28,43,57,70,82,93,103,112,120,127,133,138,139,140,141,142},{29,44,58,71,83,94,104,113,121,128,134,139,143,144,145,146},{30,45,59,72,84,95,105,114,122,129,135,140,144,147,148,149},{31,46,60,73,85,96,106,115,123,130,136,141,145,148,150,151},{32,47,61,74,86,97,107,116,124,131,137,142,146,149,151,152}},
{{18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34},{19,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50},{20,36,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65},{21,37,52,66,67,68,69,70,71,72,73,74,75,76,77,78,79},{22,38,53,67,80,81,82,83,84,85,86,87,88,89,90,91,92},{23,39,54,68,81,93,94,95,96,97,98,99,100,101,102,103,104},{24,40,55,69,82,94,105,106,107,108,109,110,111,112,113,114,115},{25,41,56,70,83,95,106,116,117,118,119,120,121,122,123,124,125},{26,42,57,71,84,96,107,117,126,127,128,129,130,131,132,133,134},{27,43,58,72,85,97,108,118,127,135,136,137,138,139,140,141,142},{28,44,59,73,86,98,109,119,128,136,143,144,145,146,147,148,149},{29,45,60,74,87,99,110,120,129,137,144,150,151,152,153,154,155},{30,46,61,75,88,100,111,121,130,138,145,151,156,157,158,159,160},{31,47,62,76,89,101,112,122,131,139,146,152,157,161,162,163,164},{32,48,63,77,90,102,113,123,132,140,147,153,158,162,165,166,167},{33,49,64,78,91,103,114,124,133,141,148,154,159,163,166,168,169},{34,50,65,79,92,104,115,125,134,142,149,155,160,164,167,169,170}},
{{19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36},{20,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53},{21,38,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69},{22,39,55,70,71,72,73,74,75,76,77,78,79,80,81,82,83,84},{23,40,56,71,85,86,87,88,89,90,91,92,93,94,95,96,97,98},{24,41,57,72,86,99,100,101,102,103,104,105,106,107,108,109,110,111},{25,42,58,73,87,100,112,113,114,115,116,117,118,119,120,121,122,123},{26,43,59,74,88,101,113,124,125,126,127,128,129,130,131,132,133,134},{27,44,60,75,89,102,114,125,135,136,137,138,139,140,141,142,143,144},{28,45,61,76,90,103,115,126,136,145,146,147,148,149,150,151,152,153},{29,46,62,77,91,104,116,127,137,146,154,155,156,157,158,159,160,161},{30,47,63,78,92,105,117,128,138,147,155,162,163,164,165,166,167,168},{31,48,64,79,93,106,118,129,139,148,156,163,169,170,171,172,173,174},{32,49,65,80,94,107,119,130,140,149,157,164,170,175,176,177,178,179},{33,50,66,81,95,108,120,131,141,150,158,165,171,176,180,181,182,183},{34,51,67,82,96,109,121,132,142,151,159,166,172,177,181,184,185,186},{35,52,68,83,97,110,122,133,143,152,160,167,173,178,182,185,187,188},{36,53,69,84,98,111,123,134,144,153,161,168,174,179,183,186,188,189}},
{{20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38},{21,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56},{22,40,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73},{23,41,58,74,75,76,77,78,79,80,81,82,83,84,85,86,87,88,89},{24,42,59,75,90,91,92,93,94,95,96,97,98,99,100,101,102,103,104},{25,43,60,76,91,105,106,107,108,109,110,111,112,113,114,115,116,117,118},{26,44,61,77,92,106,119,120,121,122,123,124,125,126,127,128,129,130,131},{27,45,62,78,93,107,120,132,133,134,135,136,137,138,139,140,141,142,143},{28,46,63,79,94,108,121,133,144,145,146,147,148,149,150,151,152,153,154},{29,47,64,80,95,109,122,134,145,155,156,157,158,159,160,161,162,163,164},{30,48,65,81,96,110,123,135,146,156,165,166,167,168,169,170,171,172,173},{31,49,66,82,97,111,124,136,147,157,166,174,175,176,177,178,179,180,181},{32,50,67,83,98,112,125,137,148,158,167,175,182,183,184,185,186,187,188},{33,51,68,84,99,113,126,138,149,159,168,176,183,189,190,191,192,193,194},{34,52,69,85,100,114,127,139,150,160,169,177,184,190,195,196,197,198,199},{35,53,70,86,101,115,128,140,151,161,170,178,185,191,196,200,201,202,203},{36,54,71,87,102,116,129,141,152,162,171,179,186,192,197,201,204,205,206},{37,55,72,88,103,117,130,142,153,163,172,180,187,193,198,202,205,207,208},{38,56,73,89,104,118,131,143,154,164,173,181,188,194,199,203,206,208,209}},
{{21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40},{22,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59},{23,42,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77},{24,43,61,78,79,80,81,82,83,84,85,86,87,88,89,90,91,92,93,94},{25,44,62,79,95,96,97,98,99,100,101,102,103,104,105,106,107,108,109,110},{26,45,63,80,96,111,112,113,114,115,116,117,118,119,120,121,122,123,124,125},{27,46,64,81,97,112,126,127,128,129,130,131,132,133,134,135,136,137,138,139},{28,47,65,82,98,113,127,140,141,142,143,144,145,146,147,148,149,150,151,152},{29,48,66,83,99,114,128,141,153,154,155,156,157,158,159,160,161,162,163,164},{30,49,67,84,100,115,129,142,154,165,166,167,168,169,170,171,172,173,174,175},{31,50,68,85,101,116,130,143,155,166,176,177,178,179,180,181,182,183,184,185},{32,51,69,86,102,117,131,144,156,167,177,186,187,188,189,190,191,192,193,194},{33,52,70,87,103,118,132,145,157,168,178,187,195,196,197,198,199,200,201,202},{34,53,71,88,104,119,133,146,158,169,179,188,196,203,204,205,206,207,208,209},{35,54,72,89,105,120,134,147,159,170,180,189,197,204,210,211,212,213,214,215},{36,55,73,90,106,121,135,148,160,171,181,190,198,205,211,216,217,218,219,220},{37,56,74,91,107,122,136,149,161,172,182,191,199,206,212,217,221,222,223,224},{38,57,75,92,108,123,137,150,162,173,183,192,200,207,213,218,222,225,226,227},{39,58,76,93,109,124,138,151,163,174,184,193,201,208,214,219,223,226,228,229},{40,59,77,94,110,125,139,152,164,175,185,194,202,209,215,220,224,227,229,230}}};

}}

//template class capd::autodiff::DagIndexer<double>;
//template class capd::autodiff::DagIndexer<long double>;

// --- End DagIndexer.cpp ---

// --- Begin include/capd/autodiff/NodeType.h ---
/////////////////////////////////////////////////////////////////////////////
/// @file NodeType.h
///
/// @author Daniel Wilczak
/////////////////////////////////////////////////////////////////////////////

// Copyright (C) 2000-2017 by the CAPD Group.
//
// This file constitutes a part of the CAPD library,
// distributed under the terms of the GNU General Public License.
// Consult  http://capd.ii.uj.edu.pl/ for details.

#ifndef _CAPD_AUTODIFF_NODETYPE_H_
#define _CAPD_AUTODIFF_NODETYPE_H_

#include <vector>
// #include "capd/autodiff/DagIndexer.h"
// #include "capd/vectalg/Multiindex.h"

namespace capd{
namespace autodiff{
/// @addtogroup autodiff
/// @{

template<class T>
struct MaskIterator{
  MaskIterator(T* data, const bool* mask) : data(data), mask(mask){}
  void operator++() { ++data; ++mask; }
  void operator++(int) { ++data; ++mask; }
  const T& operator*() const { return *data; }
  T& operator*() { return *data; }
  const T& operator[](unsigned j) const { return data[j]; }
  T& operator[](unsigned j) { return data[j]; }
  void operator+=(unsigned j) { data+=j; mask+=j; }
  MaskIterator operator+(int j) const { return MaskIterator(data+j,mask+j); }

  T* data;
  const bool* mask;
};

template<class T>
inline bool getMask(MaskIterator<T> i){
  return *(i.mask);
}

template<class T>
inline bool getMask(MaskIterator<T> i, unsigned j){
  return i.mask[j];
}

template<class T>
inline bool getMask(T*) { return true; }

template<class T>
inline bool getMask(T*, unsigned) { return true; }

// ANY change in NodeType MUST be synchronized with an array of functions file eval.hpp
enum NodeType {
// ------------------ ADD -------------------------------
    NODE_ADD                    =0, // f(x,y,..)*g(x,y,..)
    NODE_CONST_PLUS_VAR         =1, // const + f(x,y,..)
    NODE_CONST_PLUS_CONST       =2, // const1 + const2
    NODE_CONST_PLUS_TIME        =3, // const + time
    NODE_CONST_PLUS_FUNTIME     =4, // const + f(time)
    NODE_TIME_PLUS_VAR          =5, // time + f(x,y,..)
    NODE_TIME_PLUS_FUNTIME      =6, // time + f(time)
    NODE_FUNTIME_PLUS_VAR       =7, // f(time) + g(x,y,...)
    NODE_FUNTIME_PLUS_FUNTIME   =8, // f(time) + g(time)
// ------------------ SUB -------------------------------
    NODE_SUB                    =10 ,
    NODE_CONST_MINUS_CONST      =11,
    NODE_CONST_MINUS_VAR        =12,
    NODE_CONST_MINUS_TIME       =13,
    NODE_CONST_MINUS_FUNTIME    =14,
    NODE_TIME_MINUS_CONST       =15,
    NODE_TIME_MINUS_FUNTIME     =16,
    NODE_TIME_MINUS_VAR         =17,
    NODE_FUNTIME_MINUS_CONST    =18,
    NODE_FUNTIME_MINUS_TIME     =19,
    NODE_FUNTIME_MINUS_FUNTIME  =20,
    NODE_FUNTIME_MINUS_VAR      =21,
    NODE_VAR_MINUS_CONST        =22,
    NODE_VAR_MINUS_TIME         =23,
    NODE_VAR_MINUS_FUNTIME      =24,
// -------------- UNARY MINUS ---------------------------
    NODE_UNARY_MINUS            =30,
    NODE_UNARY_MINUS_CONST      =31,
    NODE_UNARY_MINUS_TIME       =32,
    NODE_UNARY_MINUS_FUNTIME    =33,
// ------------------- MUL ------------------------------
    NODE_MUL                    =40, // f(x,y,..)*g(x,y,..)
    NODE_MUL_CONST_BY_VAR       =41, // const*f(x,y,...)
    NODE_MUL_CONST_BY_CONST     =42, // const1*const2
    NODE_MUL_CONST_BY_TIME      =43, // const*t
    NODE_MUL_CONST_BY_FUNTIME   =44, // const*f(t)
    NODE_MUL_TIME_BY_VAR        =45, // t*f(x,y,...)
    NODE_MUL_TIME_BY_FUNTIME    =46, // t*f(t)
    NODE_MUL_FUNTIME_BY_VAR     =47, // f(t)*g(x,y,.....)
    NODE_MUL_FUNTIME_BY_FUNTIME =48, // f(t)*g(t)
// ------------------- DIV ------------------------------
    NODE_DIV                    =50, // Division of type const/var cannot be significantly improved.
    NODE_DIV_VAR_BY_CONST       =51, // Formulae for var/var and const/var differ by only one subtraction.
    NODE_DIV_VAR_BY_TIME        =52, // The formula depends mainly on the result and the denominator.
    NODE_DIV_VAR_BY_FUNTIME     =53, // Thus we define special nodes for these cases when denominator is simpler.
    NODE_DIV_TIME_BY_CONST      =54, // cases TIME_BY_FUNTIME and TIME_BY_VAR are covered by FUNTIME_BY_FUNTIME and FUNTIME_BY_VAR, respectively.
    NODE_DIV_FUNTIME_BY_CONST   =55,
    NODE_DIV_FUNTIME_BY_TIME    =56,
    NODE_DIV_FUNTIME_BY_FUNTIME =57,
    NODE_DIV_CONST_BY_CONST     =58,
// --------------- SQUARE AND SQUARE ROOT--------------------
    NODE_SQR                    =60,
    NODE_SQR_CONST              =61,
    NODE_SQR_TIME               =62,
    NODE_SQR_FUNTIME            =63,
    NODE_SQRT                   =64,
    NODE_SQRT_CONST             =65,
    NODE_SQRT_TIME              =66,
    NODE_SQRT_FUNTIME           =67,
// ------------------- VARIOUS POWERS -----------------------
    NODE_POW                      =70, // general power, exponent can be negative or an interval
    NODE_POW_CONST                =71,
    NODE_POW_TIME                 =72,
    NODE_POW_FUNTIME              =73,
    NODE_NATURAL_POW              =74,  // exponent is a natural number
    NODE_NATURAL_POW_CONST        =75,  // exponent is a natural number
    NODE_NATURAL_POW_TIME         =76,  // exponent is a natural number
    NODE_NATURAL_POW_FUNTIME      =77,  // exponent is a natural number
    NODE_NEG_INT_POW              =78,  // exponent is a negative integer
    NODE_NEG_INT_POW_CONST        =79,  // exponent is a negative integer
    NODE_NEG_INT_POW_TIME         =80,  // exponent is a negative integer
    NODE_NEG_INT_POW_FUNTIME      =81,  // exponent is a negative integer
    NODE_HALF_INT_POW             =82,  // exponent is of the form N/2, where N is an integer (positive or negative)
    NODE_HALF_INT_POW_CONST       =83,  // exponent is of the form N/2
    NODE_HALF_INT_POW_TIME        =84,  // exponent is of the form N/2
    NODE_HALF_INT_POW_FUNTIME     =85,  // exponent is of the form N/2
    NODE_CUBE                     =86,
    NODE_CUBE_CONST               =87,
    NODE_CUBE_TIME                =88,
    NODE_CUBE_FUNTIME             =89,
    NODE_QUARTIC                  =90,
    NODE_QUARTIC_CONST            =91,
    NODE_QUARTIC_TIME             =92,
    NODE_QUARTIC_FUNTIME          =93,
//  -------------------- EXP LOG ----------------------------
    NODE_EXP                    =100,
    NODE_EXP_CONST              =101,
    NODE_EXP_TIME               =102,
    NODE_EXP_FUNTIME            =103,
    NODE_LOG                    =104,
    NODE_LOG_CONST              =105,
    NODE_LOG_TIME               =106,
    NODE_LOG_FUNTIME            =107,
//  ------------------- SIN -----------------------------
    NODE_SIN                    =110,
    NODE_SIN_CONST              =111,
    NODE_SIN_TIME               =112,
    NODE_SIN_FUNTIME            =113,
//  ------------------- ATAN -----------------------------
    NODE_ATAN,
    NODE_ATAN_CONST,
    NODE_ATAN_TIME,
    NODE_ATAN_FUNTIME,
//  ------------------- ASIN -----------------------------
    NODE_ONE_MINUS_SQR,
    NODE_ONE_MINUS_SQR_CONST,
    NODE_ONE_MINUS_SQR_TIME,
    NODE_ONE_MINUS_SQR_FUNTIME,
    NODE_ASIN,
    NODE_ASIN_CONST,
    NODE_ASIN_TIME,
    NODE_ASIN_FUNTIME,
    NODE_ACOS,
    NODE_ACOS_CONST,
    NODE_ACOS_TIME,
    NODE_ACOS_FUNTIME,
// ---------------------- VARS, PARAMS, CONST and COS ----------------
    NODE_NULL,
    NODE_CONST,
    NODE_TIME,
    NODE_PARAM,
    NODE_VAR,
    NODE_COS
  };

struct Int4{
  int left, right, result, op;
  Int4(int _left, int _right, int _result, int _op)
    : left(_left), right(_right), result(_result), op(_op)
  {}
};

struct Node : public Int4{
  double val;
  bool isConst;
  bool isTimeDependentOnly;
  static std::vector<Node>* dag;
  Node(int left, int right, int result, int op)
    : Int4(left,right,result,op),
      val(0.0), isConst(false), isTimeDependentOnly(false)
  {}

  explicit Node(double val)
    : Int4(NODE_NULL,NODE_NULL,dag->size(),NODE_CONST),
      val(val), isConst(true),isTimeDependentOnly(false)
  {
    dag->push_back(*this);
  }

  Node()
    : Int4(NODE_NULL,NODE_NULL,0,NODE_NULL),
      val(0.), isConst(false),isTimeDependentOnly(false)
  {}
};

Node operator+(const Node& x, const Node& y);
Node operator+(const Node& x, double y);
Node operator+(double x, const Node& y);

Node operator-(const Node& x, const Node& y);
Node operator-(const Node& x, double y);
Node operator-(double x, const Node& y);

Node operator-(const Node& x);

Node operator*(const Node& x, const Node& y);
Node operator*(const Node& x, double y);
Node operator*(double x, const Node& y);

Node operator/(const Node& x, const Node& y);
Node operator/(const Node& x, double y);
Node operator/(double x, const Node& y);

Node operator^(const Node& x, double);

Node& operator+=(Node& x, const Node& y);
Node& operator+=(Node& x, double y);

Node& operator-=(Node& x, const Node& y);
Node& operator-=(Node& x, double y);

Node& operator*=(Node& x, const Node& y);
Node& operator*=(Node& x, double y);

Node& operator/=(Node& x, const Node& y);
Node& operator/=(Node& x, double y);

template<class T>
class AbstractNode{
public:
  T* left;
  T* right;
  T* result;
  DagIndexer<T>* dag;
  void setDag(DagIndexer<T>* dag){
    this->dag = dag;
  }
  AbstractNode() : left(0), right(0), result(0){}
  virtual void evalC0(const int coeffNo) = 0;
  virtual void eval(const int degree, const int coeffNo) = 0;
  virtual void eval(const int degree, const int coeffNo, const bool* mask) = 0;
  virtual void evalC0HomogenousPolynomial() = 0;
  virtual void evalHomogenousPolynomial(const int degree, const int coeffNo) = 0;
  virtual void evalHomogenousPolynomial(const int degree, const int coeffNo, const bool* mask) = 0;
  virtual const char* name() const = 0;
  virtual ~AbstractNode() {}
};

#define CAPD_MAKE_DAG_NODE(ClassName)\
  template<class T>\
  class ClassName##Node : public AbstractNode<T>\
  { \
    public:\
    void evalC0(const int coeffNo) {\
      ClassName::evalC0(this->left,this->right,this->result,coeffNo);\
    }\
    void eval(const int degree, const int coeffNo) {\
      ClassName::eval(degree,this->left,this->right,this->result,this->dag,coeffNo);\
    }\
    void eval(const int degree, const int coeffNo, const bool* mask) {\
      ClassName::eval(degree,this->left,this->right,MaskIterator<T>(this->result,mask),this->dag,coeffNo);\
    }\
    void evalC0HomogenousPolynomial() {\
      ClassName::evalC0HomogenousPolynomial(this->left,this->right,this->result);\
    }\
    void evalHomogenousPolynomial(const int degree, const int coeffNo) {\
      ClassName::evalHomogenousPolynomial(degree,this->left,this->right,this->result,this->dag,coeffNo);\
    }\
    void evalHomogenousPolynomial(const int degree, const int coeffNo, const bool* mask) {\
      ClassName::evalHomogenousPolynomial(degree,this->left,this->right,MaskIterator<T>(this->result,mask),this->dag,coeffNo);\
    }\
    const char* name() const {\
      return #ClassName;\
    }\
  }

typedef Int4 MyNode;
/// @}
}} // namespace capd::autodiff

capd::autodiff::Node sqr(const capd::autodiff::Node&);
capd::autodiff::Node sqrt(const capd::autodiff::Node&);
capd::autodiff::Node exp(const capd::autodiff::Node&);
capd::autodiff::Node log(const capd::autodiff::Node&);
capd::autodiff::Node sin(const capd::autodiff::Node&);
capd::autodiff::Node cos(const capd::autodiff::Node&);
capd::autodiff::Node atan(const capd::autodiff::Node&);
capd::autodiff::Node asin(const capd::autodiff::Node&);
capd::autodiff::Node acos(const capd::autodiff::Node&);

#endif

// --- End include/capd/autodiff/NodeType.h ---

// --- Begin NodeType.cpp ---
/// @addtogroup autodiff
/// @{

/////////////////////////////////////////////////////////////////////////////
/// @file Parser.cpp
///
/// @author Daniel Wilczak
/////////////////////////////////////////////////////////////////////////////

// Copyright (C) 2000-2012 by the CAPD Group.
//
// This file constitutes a part of the CAPD library,
// distributed under the terms of the GNU General Public License.
// Consult  http://capd.wsb-nlu.edu.pl/ for details.

#include <stdexcept>
#include <cstdlib>

// #include "capd/autodiff/NodeType.h"

// ------------------------------------------------------------------------
/// a general function for creating new binary node in DAG representing an expression
/// should not been used by a user
capd::autodiff::Node createBinaryNode(
    const capd::autodiff::Node& left,
    const capd::autodiff::Node& right,
    capd::autodiff::NodeType op
)
{
  std::vector<capd::autodiff::Node>& dag = * capd::autodiff::Node::dag;
  capd::autodiff::Node node(0,0,0,op);
  node.left = left.result;
  node.right = right.result;
  node.result = dag.size();
  node.isConst = dag[node.left].isConst and dag[node.right].isConst;
  node.isTimeDependentOnly = dag[node.left].isTimeDependentOnly and dag[node.right].isTimeDependentOnly;
  dag.push_back(node);
  return node;
}

// ------------------------------------------------------------------------
/// a general function for creating new unary node in DAG representing an expression
/// should not been used by a user
capd::autodiff::Node createUnaryNode(
    const capd::autodiff::Node& left,
    capd::autodiff::NodeType op
)
{
  std::vector<capd::autodiff::Node>& dag = * capd::autodiff::Node::dag;
  capd::autodiff::Node node(0,capd::autodiff::NODE_NULL,0,op);
  node.left = left.result;
  node.result = dag.size();
  node.isConst = dag[node.left].isConst;
  node.isTimeDependentOnly = dag[node.left].isTimeDependentOnly;
  dag.push_back(node);
  return node;
}

std::vector<capd::autodiff::Node>* capd::autodiff::Node::dag = 0;

namespace capd{
  namespace autodiff{


Node operator+(const Node& x, const Node& y){
  return createBinaryNode(x,y,NODE_ADD);
}

Node operator+(const Node& x, double y)
{
  Node c(y);
  return x+c;
}

Node operator+(double x, const Node& y){
  Node c(x);
  return c+y;
}

// ###############################################################

Node operator-(const Node& x, const Node& y){
  return createBinaryNode(x,y,NODE_SUB);
}

Node operator-(const Node& x, double y)
{
  Node c(y);
  return x-c;
}

Node operator-(double x, const Node& y){
  Node c(x);
  return c-y;
}

Node operator-(const Node& x){
  return createUnaryNode(x,NODE_UNARY_MINUS);
}

// ###############################################################

Node operator*(const Node& x, const Node& y){
  return createBinaryNode(x,y,NODE_MUL);
}

Node operator*(const Node& x, double y)
{
  Node c(y);
  return x*c;
}

Node operator*(double x, const Node& y){
  Node c(x);
  return c*y;
}

// ###############################################################

Node operator^(const Node& x, double y){
  if(y==1.)
    return x;
  if(y==0.)
    throw std::logic_error("Map constructor error: an expression of the form x^c, where c=0 is not allowed.");
  Node c(y);
  return createBinaryNode(x,c,NODE_POW);
}

// ###############################################################

Node operator/(const Node& x, const Node& y){
  return createBinaryNode(x,y,NODE_DIV);
}

Node operator/(const Node& x, double y)
{
  Node c(y);
  return x/c;
}

Node operator/(double x, const Node& y){
  Node c(x);
  return c/y;
}

// ###############################################################

Node& operator+=(Node& x, const Node& y){
  Node r=x+y;
  x = r;
  return x;
}

Node& operator+=(Node& x, double y){
  Node c(y);
  return x+=c;
}

// ###############################################################

Node& operator-=(Node& x, const Node& y){
  Node r=x-y;
  x = r;
  return x;
}

Node& operator-=(Node& x, double y){
  Node c(y);
  return x-=c;
}

// ###############################################################

Node& operator*=(Node& x, const Node& y){
  Node r=x*y;
  x = r;
  return x;
}

Node& operator*=(Node& x, double y){
  Node c(y);
  return x*=c;
}

// ###############################################################

Node& operator/=(Node& x, const Node& y){
  Node r=x/y;
  x = r;
  return x;
}

Node& operator/=(Node& x, double y){
  Node c(y);
  return x/=c;
}

}} // namespace capd::map

// ###############################################################

capd::autodiff::Node sqr(const capd::autodiff::Node& x){
  return createUnaryNode(x,capd::autodiff::NODE_SQR);
}

capd::autodiff::Node sqrt(const capd::autodiff::Node& x){
  return createUnaryNode(x,capd::autodiff::NODE_SQRT);
}

capd::autodiff::Node exp(const capd::autodiff::Node& x){
  return createUnaryNode(x,capd::autodiff::NODE_EXP);
}

capd::autodiff::Node log(const capd::autodiff::Node& x){
  return createUnaryNode(x,capd::autodiff::NODE_LOG);
}

capd::autodiff::Node sin(const capd::autodiff::Node& x){
  using namespace capd::autodiff;
  std::vector<Node>& dag = *Node::dag;
  Node c = createUnaryNode(x,NODE_COS);
  Node s = createUnaryNode(x,NODE_SIN);
  dag[c.result].right = s.result;
  dag[s.result].right = c.result;
  return s;
}

capd::autodiff::Node cos(const capd::autodiff::Node& x){
  using namespace capd::autodiff;
  std::vector<Node>& dag = *Node::dag;
  Node s = createUnaryNode(x,NODE_SIN);
  Node c = createUnaryNode(x,NODE_COS);
  dag[c.result].right = s.result;
  dag[s.result].right = c.result;
  return c;
}

capd::autodiff::Node atan(const capd::autodiff::Node& x){
  using namespace capd::autodiff;
  Node u = createUnaryNode(x,NODE_SQR);
  return createBinaryNode(x,u,NODE_ATAN);
}

capd::autodiff::Node asin(const capd::autodiff::Node& x){
  using namespace capd::autodiff;
  Node u = createUnaryNode(x,NODE_ONE_MINUS_SQR);
  Node v = createUnaryNode(u,NODE_SQRT);
  return createBinaryNode(x,v,NODE_ASIN);
}

capd::autodiff::Node acos(const capd::autodiff::Node& x){
  using namespace capd::autodiff;
  Node u = createUnaryNode(x,NODE_ONE_MINUS_SQR);
  Node v = createUnaryNode(u,NODE_SQRT);
  return createBinaryNode(x,v,NODE_ACOS);
}

/*
Node pow(const Node&, int);
Node pow(const Node&, double);
*/


/// @}

// --- End NodeType.cpp ---

// --- Begin include/capd/autodiff/EvalSub.h ---
/////////////////////////////////////////////////////////////////////////////
/// @file EvalSub.h
///
/// @author Daniel Wilczak
/////////////////////////////////////////////////////////////////////////////

// Copyright (C) 2000-2017 by the CAPD Group.
//
// This file constitutes a part of the CAPD library,
// distributed under the terms of the GNU General Public License.
// Consult  http://capd.ii.uj.edu.pl/ for details.

#ifndef _CAPD_AUTODIFF_EVAL_SUB_H_
#define _CAPD_AUTODIFF_EVAL_SUB_H_

// #include "capd/autodiff/NodeType.h"

namespace capd{
namespace autodiff{
/// @addtogroup autodiff
/// @{

// -------------------- Sub ------------------------------------

namespace Sub
{
  template<class T, class R>
  inline void evalC0(const T* left, const T* right, R result, const unsigned coeffNo)
  {
    result[coeffNo] = left[coeffNo] - right[coeffNo];
  }

  template<class T, class R>
  inline void evalC0HomogenousPolynomial(const T* left, const T* right, R result)
  {
    *result = *left - *right;
  }

  template<class T, class R>
  inline void evalHelper(const T* left, const T* right, R result, const unsigned dataSize, const unsigned order, const unsigned shift)
  {
    left += shift;
    right += shift;
    result += shift;
    const T* end = left + dataSize*order;
    for(;left!=end; left+=order,right+=order,result+=order)
      if(getMask(result))
        *result = *left - *right;
  }

  template<class T, class R>
  inline void eval(const unsigned degree, const T* left, const T* right, R result, DagIndexer<T>* dag, const unsigned coeffNo)
  {
    evalHelper(left,right,result,binomial(dag->domainDimension()+degree,degree),dag->getOrder()+1,coeffNo);
  }

  template<class T, class R>
  inline void evalHomogenousPolynomial(const unsigned degree, const T* left, const T* right, R result, DagIndexer<T>* dag, const unsigned coeffNo)
  {
    const unsigned dim = dag->domainDimension();
    const unsigned order = dag->getOrder()+1;
    const unsigned shift = binomial(dim+degree-1,dim)*order;
    const unsigned dataSize = binomial(dim+degree-1,degree);
    evalHelper(left,right,result,dataSize,order,coeffNo+shift);
  }
}

// -------------------- ConstMinusVar  --------------------------

namespace ConstMinusVar
{
  template<class T, class R>
  inline void evalC0(const T* left, const T* right, R result, const unsigned coeffNo)
  {
    if(coeffNo)
      result[coeffNo] = -right[coeffNo];
    else
      *result = *left - *right;
//    result[coeffNo] = (coeffNo!=0) ? -right[coeffNo] : *left - *right;
  }

  template<class T, class R>
  inline void evalC0HomogenousPolynomial(const T* left, const T* right, R result)
  {
    *result = *left - *right;
  }

  template<class T, class R>
  inline void evalHelper(const T* right, R result, const unsigned dataSize, const unsigned order, const unsigned shift)
  {
    right += shift;
    result += shift;
    const T* end = right + dataSize*order;
    for(;right!=end; right+=order,result+=order)
      if(getMask(result))
        *result = -(*right);
  }

  template<class T, class R>
  inline void eval(const unsigned degree, const T* left, const T* right, R result, DagIndexer<T>* dag, const unsigned coeffNo)
  {
    evalC0(left,right,result,coeffNo);
    const unsigned dim = dag->domainDimension();
    const unsigned order = dag->getOrder()+1;
    evalHelper(right,result,binomial(dim+degree,degree)-1,order,coeffNo+order);
  }

  template<class T, class R>
  inline void evalHomogenousPolynomial(const unsigned degree, const T* /*left*/, const T* right, R result, DagIndexer<T>* dag, const unsigned coeffNo){
    const unsigned dim = dag->domainDimension();
    const unsigned order = dag->getOrder()+1;
    const unsigned shift = binomial(dim+degree-1,dim)*order;
    const unsigned dataSize = binomial(dim+degree-1,degree);
    evalHelper(right,result,dataSize,order,coeffNo+shift);
  }
}

// -------------------- ConstMinusFunTime  --------------------------

namespace ConstMinusFunTime
{
  template<class T, class R>
  inline void evalC0(const T* left, const T* right, R result, const unsigned coeffNo)
  {
    if(coeffNo)
      result[coeffNo] = -right[coeffNo];
    else
      *result = *left - *right;
  }

  template<class T, class R>
  inline void evalC0HomogenousPolynomial(const T* left, const T* right, R result)
  {
    *result = *left - *right;
  }

  template<class T, class R>
  inline void eval(const unsigned /*degree*/, const T* left, const T* right, R result, DagIndexer<T>* /*dag*/, const unsigned coeffNo)
  {
    evalC0(left,right,result,coeffNo);
  }

  template<class T, class R>
  inline void evalHomogenousPolynomial(const unsigned /*degree*/, const T* /*left*/, const T* /*right*/, R /*result*/, DagIndexer<T>* /*dag*/, const unsigned /*coeffNo*/)
  {}
}

// -------------------- ConstMinusTime  --------------------------

namespace ConstMinusTime
{
  template<class T, class R>
  inline void evalC0(const T* left, const T* right, R result, const unsigned coeffNo)
  {
    if(coeffNo==1)
      result[coeffNo] = -TypeTraits<T>::one();
    else if (coeffNo==0)
      *result = *left - *right;
  }

  template<class T, class R>
  inline void evalC0HomogenousPolynomial(const T* left, const T* right, R result)
  {
    *result = *left - *right;
  }

  template<class T, class R>
  inline void eval(const unsigned /*degree*/, const T* left, const T* right, R result, DagIndexer<T>* /*dag*/, const unsigned coeffNo)
  {
    evalC0(left,right,result,coeffNo);
  }


  template<class T, class R>
  inline void evalHomogenousPolynomial(const unsigned /*degree*/, const T* /*left*/, const T* /*right*/, R /*result*/, DagIndexer<T>* /*dag*/, const unsigned /*coeffNo*/)
  {}
}

// -------------------- ConstMinusConst  --------------------------

namespace ConstMinusConst
{
  template<class T, class R>
  inline void evalC0(const T* left, const T* right, R result, const unsigned coeffNo)
  {
    if(coeffNo)
    {}
    else
      *result = *left - *right;
  }

  template<class T, class R>
  inline void evalC0HomogenousPolynomial(const T* left, const T* right, R result)
  {
    *result = *left - *right;
  }

  template<class T, class R>
  inline void eval(const unsigned /*degree*/, const T* left, const T* right, R result, DagIndexer<T>* /*dag*/, const unsigned coeffNo)
  {
    evalC0(left,right,result,coeffNo);
  }

  template<class T, class R>
  inline void evalHomogenousPolynomial(const unsigned /*degree*/, const T* /*left*/, const T* /*right*/, R /*result*/, DagIndexer<T>* /*dag*/, const unsigned /*coeffNo*/)
  {}
}

// -------------------- TimeMinusConst  --------------------------

namespace TimeMinusConst
{
  template<class T, class R>
  inline void evalC0(const T* left, const T* right, R result, const unsigned coeffNo)
  {
    if(coeffNo==1)
      result[coeffNo] = TypeTraits<T>::one();
    else if (coeffNo==0)
      *result = *left - *right;
  }

  template<class T, class R>
  inline void evalC0HomogenousPolynomial(const T* left, const T* right, R result)
  {
    *result = *left - *right;
  }

  template<class T, class R>
  inline void eval(const unsigned /*degree*/, const T* left, const T* right, R result, DagIndexer<T>* /*dag*/, const unsigned coeffNo){
    evalC0(left,right,result,coeffNo);
  }

  template<class T, class R>
  inline void evalHomogenousPolynomial(const unsigned /*degree*/, const T* /*left*/, const T* /*right*/, R /*result*/, DagIndexer<T>* /*dag*/, const unsigned /*coeffNo*/)
  {}
}

// -------------------- TimeMinusFunTime  --------------------------

namespace TimeMinusFunTime
{
  template<class T, class R>
  inline void evalC0(const T* left, const T* right, R result, const unsigned coeffNo)
  {
    if(coeffNo>1)
      result[coeffNo] = - right[coeffNo];
    else
      result[coeffNo] = left[coeffNo] - right[coeffNo];
  }

  template<class T, class R>
  inline void evalC0HomogenousPolynomial(const T* left, const T* right, R result)
  {
    *result = *left - *right;
  }

  template<class T, class R>
  inline void eval(const unsigned /*degree*/, const T* left, const T* right, R result, DagIndexer<T>* /*dag*/, const unsigned coeffNo)
  {
    evalC0(left,right,result,coeffNo);
  }

  template<class T, class R>
  inline void evalHomogenousPolynomial(const unsigned /*degree*/, const T* /*left*/, const T* /*right*/, R /*result*/, DagIndexer<T>* /*dag*/, const unsigned /*coeffNo*/)
  {}
}

// -------------------- TimeMinusVar  --------------------------

namespace TimeMinusVar
{
  template<class T, class R>
  inline void evalC0(const T* left, const T* right, R result, const unsigned coeffNo)
  {
    TimeMinusFunTime::evalC0(left,right,result,coeffNo);
  }

  template<class T, class R>
  inline void evalC0HomogenousPolynomial(const T* left, const T* right, R result)
  {
    *result = *left - *right;
  }

  template<class T, class R>
  inline void eval(const unsigned degree, const T* left, const T* right, R result, DagIndexer<T>* dag, const unsigned coeffNo)
  {
    evalC0(left,right,result,coeffNo);
    const unsigned dim = dag->domainDimension();
    const unsigned order = dag->getOrder()+1;
    ConstMinusVar::evalHelper(right,result,binomial(dim+degree,degree)-1,order,coeffNo+order);
  }

  template<class T, class R>
  inline void evalHomogenousPolynomial(const unsigned degree, const T* left, const T* right, R result, DagIndexer<T>* dag, const unsigned coeffNo)
  {
    ConstMinusVar::evalHomogenousPolynomial(degree,left,right,result,dag,coeffNo);
  }
}

// -------------------- FunTimeMinusVar  --------------------------

namespace FunTimeMinusVar
{
  template<class T, class R>
  inline void evalC0(const T* left, const T* right, R result, const unsigned coeffNo)
  {
    result[coeffNo] = left[coeffNo] - right[coeffNo];
  }

  template<class T, class R>
  inline void evalC0HomogenousPolynomial(const T* left, const T* right, R result)
  {
    *result = *left - *right;
  }

  template<class T,class R>
  inline void eval(const unsigned degree, const T* left, const T* right, R result, DagIndexer<T>* dag, const unsigned coeffNo)
  {
    evalC0(left,right,result,coeffNo);
    const unsigned dim = dag->domainDimension();
    const unsigned order = dag->getOrder()+1;
    ConstMinusVar::evalHelper(right,result,binomial(dim+degree,degree)-1,order,coeffNo+order);
  }

  template<class T, class R>
  inline void evalHomogenousPolynomial(const unsigned degree, const T* left, const T* right, R result, DagIndexer<T>* dag, const unsigned coeffNo)
  {
    ConstMinusVar::evalHomogenousPolynomial(degree,left,right,result,dag,coeffNo);
  }
}

// -------------------- FunTimeMinusFunTime  --------------------------

namespace FunTimeMinusFunTime
{
  template<class T, class R>
  inline void evalC0(const T* left, const T* right, R result, const unsigned coeffNo)
  {
    result[coeffNo] = left[coeffNo] - right[coeffNo];
  }

  template<class T, class R>
  inline void evalC0HomogenousPolynomial(const T* left, const T* right, R result)
  {
    *result = *left - *right;
  }

  template<class T, class R>
  inline void eval(const unsigned /*degree*/, const T* left, const T* right, R result, DagIndexer<T>* /*dag*/, const unsigned coeffNo)
  {
    result[coeffNo] = left[coeffNo] - right[coeffNo];
  }

  template<class T, class R>
  inline void evalHomogenousPolynomial(const unsigned /*degree*/, const T* /*left*/, const T* /*right*/, R /*result*/, DagIndexer<T>* /*dag*/, const unsigned /*coeffNo*/)
  {}
}

// -------------------- FunTimeMinusTime  --------------------------

namespace FunTimeMinusTime
{
  template<class T, class R>
  inline void evalC0(const T* left, const T* right, R result, const unsigned coeffNo)
  {
    if(coeffNo>1)
      result[coeffNo] = left[coeffNo];
    else
      result[coeffNo] = left[coeffNo] - right[coeffNo];
  }

  template<class T, class R>
  inline void evalC0HomogenousPolynomial(const T* left, const T* right, R result)
  {
    *result = *left - *right;
  }

  template<class T, class R>
  inline void eval(const unsigned /*degree*/, const T* left, const T* right, R result, DagIndexer<T>* /*dag*/, const unsigned coeffNo)
  {
    evalC0(left,right,result,coeffNo);
  }

  template<class T, class R>
  inline void evalHomogenousPolynomial(const unsigned /*degree*/, const T* /*left*/, const T* /*right*/, R /*result*/, DagIndexer<T>* /*dag*/, const unsigned /*coeffNo*/)
  {}
}

// -------------------- FunTimeMinusConst  --------------------------

namespace FunTimeMinusConst
{
  template<class T, class R>
  inline void evalC0(const T* left, const T* right, R result, const unsigned coeffNo)
  {
    if(coeffNo)
      result[coeffNo] = left[coeffNo];
    else
      *result = *left - *right;
  }

  template<class T, class R>
  inline void evalC0HomogenousPolynomial(const T* left, const T* right, R result)
  {
    *result = *left - *right;
  }

  template<class T, class R>
  inline void eval(const unsigned /*degree*/, const T* left, const T* right, R result, DagIndexer<T>* /*dag*/, const unsigned coeffNo){
    evalC0(left,right,result,coeffNo);
  }

  template<class T, class R>
  inline void evalHomogenousPolynomial(const unsigned /*degree*/, const T* /*left*/, const T* /*right*/, R /*result*/, DagIndexer<T>* /*dag*/, const unsigned /*coeffNo*/)
  {}
}

// -------------------- VarMinusConst --------------------------

namespace VarMinusConst
{
  template<class T, class R>
  inline void evalC0(const T* left, const T* right, R result, const unsigned coeffNo)
  {
    if(coeffNo)
      result[coeffNo] = left[coeffNo];
    else
      *result = *left - *right;
  }

  template<class T, class R>
  inline void evalC0HomogenousPolynomial(const T* left, const T* right, R result)
  {
    *result = *left - *right;
  }

  template<class T, class R>
  inline void evalHelper(const T* left, R result, const unsigned dataSize, const unsigned order, const unsigned shift)
  {
    left += shift;
    result += shift;
    const T* end = left + dataSize*order;
    for(;left!=end; left+=order,result+=order)
      if(getMask(result))
        *result = *left;
  }

  template<class T, class R>
  inline void eval(const unsigned degree, const T* left, const T* right, R result, DagIndexer<T>* dag, const unsigned coeffNo)
  {
    evalC0(left,right,result,coeffNo);
    const unsigned dim = dag->domainDimension();
    const unsigned order = dag->getOrder()+1;
    evalHelper(left,result,binomial(dim+degree,degree)-1,order,coeffNo+order);
  }

  template<class T, class R>
  inline void evalHomogenousPolynomial(const unsigned degree, const T* left, const T* /*right*/, R result, DagIndexer<T>* dag, const unsigned coeffNo){
    const unsigned dim = dag->domainDimension();
    const unsigned order = dag->getOrder()+1;
    const unsigned shift = binomial(dim+degree-1,dim)*order;
    const unsigned dataSize = binomial(dim+degree-1,degree);
    evalHelper(left,result,dataSize,order,coeffNo+shift);
  }
}

// -------------------- VarMinusFunTime  --------------------------

namespace VarMinusFunTime
{
  template<class T, class R>
  inline void evalC0(const T* left, const T* right, R result, const unsigned coeffNo)
  {
    result[coeffNo] = left[coeffNo] - right[coeffNo];
  }

  template<class T, class R>
  inline void evalC0HomogenousPolynomial(const T* left, const T* right, R result)
  {
    *result = *left - *right;
  }

  template<class T, class R>
  inline void eval(const unsigned degree, const T* left, const T* right, R result, DagIndexer<T>* dag, const unsigned coeffNo)
  {
    evalC0(left,right,result,coeffNo);
    const unsigned dim = dag->domainDimension();
    const unsigned order = dag->getOrder()+1;
    VarMinusConst::evalHelper(left,result,binomial(dim+degree,degree)-1,order,coeffNo+order);
  }

  template<class T, class R>
  inline void evalHomogenousPolynomial(const unsigned degree, const T* left, const T* right, R result, DagIndexer<T>* dag, const unsigned coeffNo)
  {
    VarMinusConst::evalHomogenousPolynomial(degree,left,right,result,dag,coeffNo);
  }
}

// -------------------- VarMinusTime  --------------------------

namespace VarMinusTime
{
  template<class T, class R>
  inline void evalC0(const T* left, const T* right, R result, const unsigned coeffNo)
  {
    FunTimeMinusTime::evalC0(left,right,result,coeffNo);
  }

  template<class T, class R>
  inline void evalC0HomogenousPolynomial(const T* left, const T* right, R result)
  {
    *result = *left - *right;
  }

  template<class T, class R>
  inline void eval(const unsigned degree, const T* left, const T* right, R result, DagIndexer<T>* dag, const unsigned coeffNo)
  {
    evalC0(left,right,result,coeffNo);
    const unsigned dim = dag->domainDimension();
    const unsigned order = dag->getOrder()+1;
    VarMinusConst::evalHelper(left,result,binomial(dim+degree,degree)-1,order,coeffNo+order);
  }

  template<class T, class R>
  inline void evalHomogenousPolynomial(const unsigned degree, const T* left, const T* right, R result, DagIndexer<T>* dag, const unsigned coeffNo)
  {
    VarMinusConst::evalHomogenousPolynomial(degree,left,right,result,dag,coeffNo);
  }
}

// ----------------------------------------------------------------------------------

//use macro to define classes/*

CAPD_MAKE_DAG_NODE(Sub);
CAPD_MAKE_DAG_NODE(ConstMinusVar);
CAPD_MAKE_DAG_NODE(ConstMinusFunTime);
CAPD_MAKE_DAG_NODE(ConstMinusTime);
CAPD_MAKE_DAG_NODE(ConstMinusConst);
CAPD_MAKE_DAG_NODE(TimeMinusConst);
CAPD_MAKE_DAG_NODE(TimeMinusFunTime);
CAPD_MAKE_DAG_NODE(TimeMinusVar);
CAPD_MAKE_DAG_NODE(FunTimeMinusConst);
CAPD_MAKE_DAG_NODE(FunTimeMinusTime);
CAPD_MAKE_DAG_NODE(FunTimeMinusFunTime);
CAPD_MAKE_DAG_NODE(FunTimeMinusVar);
CAPD_MAKE_DAG_NODE(VarMinusConst);
CAPD_MAKE_DAG_NODE(VarMinusTime);
CAPD_MAKE_DAG_NODE(VarMinusFunTime);
/// @}
}} // namespace capd::autodiff

#endif

// --- End include/capd/autodiff/EvalSub.h ---


// --- Main Function ---

using namespace capd;
using namespace capd::autodiff;
using namespace capd::vectalg;

int main() {
    const int order = 5;
    const int num_nodes = 50;
    const int size_per_node = order + 1;

    // Simulate memory layout from DagIndexer
    std::vector<double> memory(num_nodes * size_per_node);
    for(size_t i=0; i<memory.size(); ++i) memory[i] = 0.0;

    double* c = &memory[0 * size_per_node];
    double* y = &memory[1 * size_per_node];
    double* z = &memory[2 * size_per_node]; // z = 3 - y

    c[0] = 3.0;
    y[0] = 2.0;

    // Use the real DagIndexer
    DagIndexer<double> dag;
    dag.resize(1, 1, 1, num_nodes, order);

    // Create ConstMinusVarNode manually
    // We need to use CAPD_MAKE_DAG_NODE created classes
    // ConstMinusVarNode is defined by macro in EvalSub.h (at the end)

    ConstMinusVarNode<double>* node = new ConstMinusVarNode<double>();
    node->left = c;
    node->right = y;
    node->result = z;
    node->setDag(&dag);

    // Solver loop
    for (int k = 0; k <= order; ++k) {
        // Evaluate equations
        node->evalC0(k);

        if (k < order) {
             y[k+1] = z[k] / (k+1);
        }
    }

    for(int i=0; i<=order; ++i) {
        std::cout << "{" << y[i] << "}" << std::endl;
    }

    delete node;

    return 0;
}
