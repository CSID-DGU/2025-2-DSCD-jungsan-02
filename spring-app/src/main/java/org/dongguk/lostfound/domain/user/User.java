package org.dongguk.lostfound.domain.user;

import jakarta.persistence.*;
import lombok.AccessLevel;
import lombok.Builder;
import lombok.Getter;
import lombok.NoArgsConstructor;

@Entity
@Getter
@NoArgsConstructor(access = AccessLevel.PROTECTED)
@Table(name = "users")
public class User {
    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    @Column(name = "login_id", nullable = false, unique = true)
    private String loginId;

    @Column(nullable = false)
    private String password;

    @Builder(access = AccessLevel.PRIVATE)
    public User(String loginId, String password) {
        this.loginId = loginId;
        this.password = password;
    }

    public static User create(String loginId, String password) {
        return User.builder()
                .loginId(loginId)
                .password(password)
                .build();
    }
}
