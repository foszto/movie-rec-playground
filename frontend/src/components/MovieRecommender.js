import React, { useState } from 'react';
import { Loader2, Star, StarHalf, Search, Calendar, Clock } from 'lucide-react';

const MovieRecommender = () => {
  const [userId, setUserId] = useState('');
  const [selectedUser, setSelectedUser] = useState(null);
  const [userHistory, setUserHistory] = useState([]);
  const [recommendations, setRecommendations] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleSearch = async (e) => {
    e.preventDefault();
    if (!userId.trim()) return;

    setLoading(true);
    setError(null);
    
    try {
      const response = await fetch(`/api/recommendations/${userId}`);
      if (!response.ok) {
        if (response.status === 404) {
          throw new Error('Felhasználó nem található');
        }
        throw new Error('Hiba történt az ajánlások betöltésekor');
      }
      const data = await response.json();
      
      setSelectedUser(userId);
      setUserHistory(data.userHistory);
      setRecommendations(data.recommendations);
    } catch (error) {
      setError(error.message);
      console.error('Error:', error);
    } finally {
      setLoading(false);
    }
  };

  const RatingStars = ({ rating }) => {
    const fullStars = Math.floor(rating);
    const hasHalfStar = rating % 1 >= 0.5;
    
    return (
      <div className="flex items-center space-x-1">
        {[...Array(fullStars)].map((_, i) => (
          <Star key={i} className="w-5 h-5 text-yellow-400 fill-yellow-400" />
        ))}
        {hasHalfStar && <StarHalf className="w-5 h-5 text-yellow-400 fill-yellow-400" />}
        <span className="ml-2 text-sm font-medium text-gray-600">({rating.toFixed(1)})</span>
      </div>
    );
  };

  const MovieBox = ({ movie, type }) => (
    <div className="group relative overflow-hidden rounded-xl bg-gradient-to-br from-white to-gray-50 shadow-md hover:shadow-xl transition-all duration-300 p-6">
      {/* Háttér dekoráció */}
      <div className="absolute top-0 right-0 w-32 h-32 bg-gradient-to-br from-blue-50 to-transparent rounded-bl-full opacity-30 transition-transform group-hover:scale-150 duration-500"></div>
      
      {/* Film címe és típus jelző */}
      <div className="relative">
        <div className="flex items-start justify-between mb-3">
          <h4 className="font-bold text-xl text-gray-800 group-hover:text-blue-600 transition-colors">
            {movie.title}
          </h4>
          {type === 'recommendation' && (
            <span className="px-3 py-1 bg-blue-100 text-blue-600 text-xs font-semibold rounded-full">
              Ajánlott
            </span>
          )}
        </div>

        {/* Műfajok */}
        <div className="flex flex-wrap gap-2 mb-4">
          {movie.genres.map((genre, index) => (
            <span 
              key={index}
              className="px-2 py-1 bg-gray-100 text-gray-600 text-xs rounded-lg hover:bg-gray-200 transition-colors"
            >
              {genre}
            </span>
          ))}
        </div>

        {/* Értékelés és időbélyeg */}
        <div className="mt-4 pt-4 border-t border-gray-100 flex items-center justify-between">
          <RatingStars rating={type === 'recommendation' ? movie.predictedRating : movie.rating} />
          
          {type === 'history' && (
            <div className="flex items-center text-gray-500">
              <Calendar className="w-4 h-4 mr-1" />
              <span className="text-sm">
                {new Date(movie.timestamp).toLocaleDateString()}
              </span>
            </div>
          )}
        </div>
      </div>
    </div>
  );

  return (
    <div className="max-w-6xl mx-auto p-6 bg-gray-50">
      <div className="text-center mb-12">
        <h1 className="text-4xl font-bold mb-4 bg-gradient-to-r from-blue-600 to-blue-800 bg-clip-text text-transparent">
          MovieLens Film Ajánló
        </h1>
        <div className="max-w-md mx-auto bg-white rounded-2xl shadow-lg p-8 border border-gray-100">
          <form onSubmit={handleSearch} className="flex flex-col sm:flex-row gap-4">
            <div className="flex-grow">
              <div className="relative">
                <input
                  type="number"
                  value={userId}
                  onChange={(e) => setUserId(e.target.value)}
                  placeholder="Adja meg a felhasználó ID-t..."
                  className="w-full pl-4 pr-10 py-3 border-2 border-gray-200 rounded-lg 
                           focus:outline-none focus:border-blue-500 transition-colors
                           text-gray-800 placeholder-gray-400
                           bg-gray-50 hover:bg-white focus:bg-white"
                  min="1"
                />
                <div className="absolute inset-y-0 right-3 flex items-center pointer-events-none">
                  <span className="text-gray-400">ID</span>
                </div>
              </div>
            </div>
            <button
              type="submit"
              disabled={loading}
              className="px-6 py-3 bg-gradient-to-r from-blue-600 to-blue-700 
                       text-white rounded-lg hover:from-blue-700 hover:to-blue-800 
                       disabled:from-blue-300 disabled:to-blue-400
                       transform transition-all duration-200 hover:scale-105
                       flex items-center justify-center gap-2 shadow-md
                       min-w-[120px]"
            >
              {loading ? (
                <Loader2 className="w-5 h-5 animate-spin" />
              ) : (
                <>
                  <Search className="w-5 h-5" />
                  <span>Keresés</span>
                </>
              )}
            </button>
          </form>
        </div>
      </div>

      {error && (
        <div className="p-4 mb-6 bg-red-50 text-red-600 rounded-lg border border-red-200 shadow-sm">
          {error}
        </div>
      )}

      {loading && (
        <div className="flex justify-center items-center py-12">
          <Loader2 className="w-12 h-12 animate-spin text-blue-600" />
        </div>
      )}

      {selectedUser && !loading && (
        <div className="space-y-12">
          <div className="rounded-2xl p-8 bg-white shadow-lg border border-gray-100">
            <div className="flex items-center justify-between mb-6">
              <h2 className="text-2xl font-bold text-gray-800">Személyre szabott ajánlások</h2>
              <span className="px-4 py-2 bg-blue-50 text-blue-600 rounded-full text-sm font-medium">
                {recommendations.length} film
              </span>
            </div>
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
              {recommendations.map(movie => (
                <MovieBox key={movie.id} movie={movie} type="recommendation" />
              ))}
            </div>
          </div>

          <div className="rounded-2xl p-8 bg-white shadow-lg border border-gray-100">
            <div className="flex items-center justify-between mb-6">
              <h2 className="text-2xl font-bold text-gray-800">Korábbi értékelések</h2>
              <span className="px-4 py-2 bg-gray-50 text-gray-600 rounded-full text-sm font-medium">
                {userHistory.length} film
              </span>
            </div>
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
              {userHistory.map(movie => (
                <MovieBox key={movie.id} movie={movie} type="history" />
              ))}
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default MovieRecommender;