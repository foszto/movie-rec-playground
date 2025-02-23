import React, { useState } from "react";
import {
  Loader2,
  Star,
  StarHalf,
  Search,
  Clock,
  Award,
  TrendingUp,
} from "lucide-react";

const MovieRecommender = () => {
  const [userId, setUserId] = useState("");
  const [selectedUser, setSelectedUser] = useState(null);
  const [userData, setUserData] = useState(null);
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
          throw new Error("User not found");
        }
        throw new Error("Error fetching data");
      }
      const data = await response.json();

      setSelectedUser(userId);
      setUserData(data);
    } catch (error) {
      setError(error.message);
      console.error("Error:", error);
    } finally {
      setLoading(false);
    }
  };

  const RatingStars = ({ rating }) => {
    const fullStars = Math.floor(rating);
    const hasHalfStar = rating % 1 >= 0.5;

    return (
      <div className="flex items-center">
        {[...Array(fullStars)].map((_, i) => (
          <Star key={i} className="w-4 h-4 text-yellow-400 fill-yellow-400" />
        ))}
        {hasHalfStar && (
          <StarHalf className="w-4 h-4 text-yellow-400 fill-yellow-400" />
        )}
        <span className="ml-1 text-sm">({rating.toFixed(1)})</span>
      </div>
    );
  };

  const MovieBox = ({ movie, type }) => (
    <div className="flex flex-col border rounded-lg bg-white p-4 shadow-sm hover:shadow-md transition-shadow">
      <h4 className="font-medium text-lg mb-2">{movie.title}</h4>
      <div className="text-sm text-gray-600 mb-2">
        {movie.genres.join(", ")}
      </div>
      <div className="mt-auto space-y-2">
        <div className="flex items-center justify-between">
          <RatingStars
            rating={
              type === "recommendation" ? movie.predictedRating : movie.rating
            }
          />
          {type === "history" && (
            <span className="text-xs text-gray-500">
              {new Date(movie.timestamp).toLocaleDateString()}
            </span>
          )}
        </div>
        {movie.reason && (
          <p className="text-sm text-gray-600 mt-2 border-t pt-2">
            {movie.reason}
          </p>
        )}
      </div>
    </div>
  );

  const UserProfile = ({ profile }) => (
    <div className="bg-white rounded-xl shadow-sm p-6 border border-gray-100 mb-8">
      <h2 className="text-xl font-semibold mb-4">User profile</h2>
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <div>
          <h3 className="font-medium mb-2">Favorite genres</h3>
          <div className="flex flex-wrap gap-2">
            {profile.favoriteGenres.map((genre) => (
              <span
                key={genre}
                className="px-3 py-1 bg-blue-50 text-blue-700 rounded-full text-sm"
              >
                {genre}
              </span>
            ))}
          </div>
        </div>
        <div>
          <h3 className="font-medium mb-2">Statistics</h3>
          <div className="space-y-2 text-sm">
            <div className="flex items-center justify-between">
              <span>Average rating:</span>
              <span className="font-medium">
                {profile.averageRating.toFixed(1)}
              </span>
            </div>
            <div className="flex items-center justify-between">
              <span>Total ratings:</span>
              <span className="font-medium">{profile.totalRatings}</span>
            </div>
            <div className="flex items-center justify-between">
              <span>Activity:</span>
              <span
                className={`font-medium ${profile.recentActivity ? "text-green-600" : "text-gray-500"}`}
              >
                {profile.recentActivity ? "Aktív" : "Inaktív"}
              </span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );

  const MovieSection = ({ title, movies, type, icon: Icon }) => (
    <div className="border rounded-xl p-6 bg-white shadow-sm mb-6">
      <div className="flex items-center gap-2 mb-4">
        <Icon className="w-5 h-5 text-blue-600" />
        <h2 className="text-xl font-semibold">{title}</h2>
        <span className="text-sm text-gray-600 ml-auto">
          ({movies.length} movie)
        </span>
      </div>
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {movies.map((movie) => (
          <MovieBox key={movie.id} movie={movie} type={type} />
        ))}
      </div>
    </div>
  );

  return (
    <div className="min-h-screen bg-gray-50 py-8">
      <div className="max-w-6xl mx-auto px-4">
        <div className="text-center mb-8">
          <h1 className="text-3xl font-bold mb-8 text-gray-800">
            MovieLens Movie Recommendation
          </h1>
          <div className="max-w-md mx-auto bg-white rounded-xl shadow-md p-6 border border-gray-100">
            <form
              onSubmit={handleSearch}
              className="flex flex-col sm:flex-row gap-4"
            >
              <div className="flex-grow">
                <div className="relative">
                  <input
                    type="number"
                    value={userId}
                    onChange={(e) => setUserId(e.target.value)}
                    placeholder="Enter the user ID... 1, 2, 3, etc."
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
                         flex items-center justify-center gap-2 shadow-sm
                         min-w-[120px]"
              >
                {loading ? (
                  <Loader2 className="w-5 h-5 animate-spin" />
                ) : (
                  <>
                    <Search className="w-5 h-5" />
                    <span>Search</span>
                  </>
                )}
              </button>
            </form>
          </div>
        </div>

        {error && (
          <div className="p-4 mb-6 bg-red-50 text-red-600 rounded-lg border border-red-200">
            {error}
          </div>
        )}

        {loading && (
          <div className="flex justify-center items-center py-12">
            <Loader2 className="w-10 h-10 animate-spin text-blue-600" />
          </div>
        )}

        {userData && !loading && (
          <>
            <UserProfile profile={userData.userProfile} />

            <MovieSection
              title="Personalised recommendations"
              movies={userData.recommendations}
              type="recommendation"
              icon={TrendingUp}
            />

            <MovieSection
              title="Top rated films by user"
              movies={userData.topRatedMovies}
              type="history"
              icon={Award}
            />

            <MovieSection
              title="Recent favourites"
              movies={userData.recentFavorites}
              type="history"
              icon={Clock}
            />
          </>
        )}
      </div>
    </div>
  );
};

export default MovieRecommender;
